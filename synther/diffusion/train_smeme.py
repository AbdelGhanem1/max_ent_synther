# synther/diffusion/train_smeme.py

import argparse
import pathlib
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import gin
import wandb
from dataclasses import dataclass, field

# Existing infrastructure
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from just_d4rl import d4rl_offline_dataset

# FIX: Import SimpleDiffusionGenerator to register it with GIN
from synther.diffusion.train_diffuser import SimpleDiffusionGenerator

# The SOTA Solvers
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from synther.diffusion.smeme_solver import SMEMESolver
from synther.diffusion.diffusion_config import edm_global_config

@gin.configurable
@dataclass
class AdjointMatchingConfig:
    """Configuration corresponding to Paper Appendix H details"""
    num_train_timesteps: int = 1000      # Discretization of the ODE
    num_inference_steps: int = 20        # K in the paper
    num_timesteps_to_load: int = 40      # Batch size for stratified sampling
    per_sample_threshold_quantile: float = 0.9 # For EMA clipping
    reward_multiplier: float = 1.0       # Will be overwritten by 1/alpha
    eta: float = 1.0                     # Sampling stochasticity (1.0 = full noise)

@dataclass
class SMEMEConfig:
    num_smeme_iterations: int = 3
    # Decreasing regularization schedule (Alpha)
    alpha_schedule: tuple = (1.0, 0.5, 0.1) 
    am_config: AdjointMatchingConfig = field(default_factory=AdjointMatchingConfig)

class DiffusionModelAdapter(nn.Module):
    def __init__(self, elucidated_model: ElucidatedDiffusion, solver_config: AdjointMatchingConfig):
        super().__init__()
        self.model = elucidated_model
        self.config = solver_config
        
        # --- ROBUST KARRAS SCHEDULE GENERATION ---
        # 1. Create the ramp (0 to 1)
        ramp = torch.linspace(0, 1, self.config.num_train_timesteps)
        
        # 2. Get Physics Parameters from Source of Truth
        sigma_min = edm_global_config.sigma_min
        sigma_max = edm_global_config.sigma_max
        rho = edm_global_config.rho
        
        # 3. Apply Karras Formula
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        
        # 4. [CRITICAL] Flip to match Solver Direction
        # Solver: t=1000 (Start) -> t=0 (End)
        # Array:  Index 0 (Max Sigma) -> Index 1000 (Min Sigma)
        # We flip so Index 1000 contains Max Sigma.
        self.register_buffer('sigmas', torch.flip(sigmas, [0]))
        
    def get_sigma_at_step(self, t_idx_tensor):
        """
        Translates solver index t (e.g. 999) -> Exact Sigma (e.g. 80.0).
        """
        # Clamp for safety
        t_idx_tensor = t_idx_tensor.long().clamp(0, len(self.sigmas) - 1)
        return self.sigmas[t_idx_tensor]

    def forward(self, x, t_idx, prompt_emb=None):
        # 1. Get sigma
        sigmas = self.get_sigma_at_step(t_idx)
        sigmas = sigmas.view(x.shape[0])
        
        # 2. Run Model (Denoising)
        denoised_x = self.model.preconditioned_network_forward(x, sigmas, clamp=False)
        
        # 3. Convert to Epsilon
        sigmas_broad = sigmas.view(-1, *([1] * (x.ndim - 1)))
        pred_epsilon = (x - denoised_x) / (sigmas_broad + 1e-5)
        
        return pred_epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="D4RL dataset name")
    parser.add_argument('--load_checkpoint', type=str, required=True, help="Path to pre-trained .pt file")
    parser.add_argument('--results_folder', type=str, default='./results_smeme')
    
    # S-MEME Hyperparameters
    parser.add_argument('--iterations', type=int, default=3, help="Number of S-MEME outer loops")
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.0, 0.5, 0.1], help="Regularization schedule")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--steps_per_iter', type=int, default=1000, help="Gradient steps per S-MEME iteration")
    
    # Configs for Model
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    parser.add_argument('--seed', type=int, default=0)
    
    # WANDB args
    parser.add_argument('--wandb-project', type=str, default="smeme-finetuning")
    
    args = parser.parse_args()

    # 1. Setup
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading
    print(f"Loading dataset info for {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    rew = d4rl_dataset['rewards'][:, None] if len(d4rl_dataset['rewards'].shape) == 1 else d4rl_dataset['rewards']
    term = d4rl_dataset['terminals'][:, None] if len(d4rl_dataset['terminals'].shape) == 1 else d4rl_dataset['terminals']
    next_obs = d4rl_dataset['next_observations']
    
    inputs_np = np.concatenate([obs, act, rew, next_obs, term], axis=1)
    input_dim = inputs_np.shape[1]
    print(f"Data Dimension: {input_dim}")

    # 3. Model Construction & Loading
    full_dataset_tensor = torch.from_numpy(inputs_np).float()
    base_model_edm = construct_diffusion_model(inputs=full_dataset_tensor)
    
    print(f"Loading weights from {args.load_checkpoint}...")
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    base_model_edm.load_state_dict(state_dict)
    base_model_edm.to(device)
    base_model_edm.eval() 

    # 4. Wrap the Model
    am_config = AdjointMatchingConfig()
    wrapped_model = DiffusionModelAdapter(base_model_edm, am_config).to(device)
    
    # 5. Configure S-MEME
    smeme_config = SMEMEConfig(
        num_smeme_iterations=args.iterations,
        alpha_schedule=tuple(args.alphas),
        am_config=am_config
    )
    
    # 6. Initialize Solver
    solver = SMEMESolver(base_model=wrapped_model, config=smeme_config)
    
    # [CRITICAL FIX] Override Optimizer with Lower LR for Stability
    # The default 1e-4 causes explosions on delicate manifolds. 1e-5 is safer.
    solver.optimizer = torch.optim.AdamW(solver.current_model.parameters(), lr=1e-5, weight_decay=1e-2)
    
    # 7. Training
    def noise_generator():
        while True:
            yield torch.randn(args.batch_size, input_dim).to(device)
            
    class InfiniteNoiseLoader:
        def __init__(self, generator, limit):
            self.gen = generator
            self.limit = limit
            self.cnt = 0
            
        def __iter__(self):
            self.cnt = 0
            return self
            
        def __next__(self):
            if self.cnt >= self.limit:
                raise StopIteration
            self.cnt += 1
            return next(self.gen)
            
    train_loader = InfiniteNoiseLoader(noise_generator(), args.steps_per_iter)
    
    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=args)

    print("Starting S-MEME Training...")
    finetuned_wrapper = solver.train(train_loader)
    
    # 8. Save Results
    # FIX: Save the underlying EDM, not the wrapper
    finetuned_edm = finetuned_wrapper.model 
    
    save_path = results_folder / "smeme_finetuned.pt"
    torch.save({
        'model': finetuned_edm.state_dict(),
        'config': smeme_config, 
        'steps': args.steps_per_iter * args.iterations
    }, save_path)
    
    print(f"Model saved to {save_path}")
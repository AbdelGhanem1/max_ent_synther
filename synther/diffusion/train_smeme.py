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
from dataclasses import dataclass

# Existing infrastructure
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from just_d4rl import d4rl_offline_dataset

# The SOTA Solvers we built
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from synther.diffusion.smeme_solver import SMEMESolver

@dataclass
class AdjointMatchingConfig:
    """Configuration corresponding to Paper Appendix H details"""
    num_train_timesteps: int = 1000      # Discretization of the ODE
    num_inference_steps: int = 40        # K in the paper (step size h = 1/K)
    num_timesteps_to_load: int = 40      # Batch size for stratified sampling
    per_sample_threshold_quantile: float = 0.9 # For EMA clipping
    reward_multiplier: float = 1.0       # Will be overwritten by 1/alpha
    eta: float = 1.0                     # Sampling stochasticity (1.0 = full noise)

@dataclass
class SMEMEConfig:
    num_smeme_iterations: int = 3
    # Decreasing regularization schedule (Alpha)
    # High alpha = close to base model. Low alpha = maximize entropy.
    alpha_schedule: tuple = (1.0, 0.5, 0.1) 
    am_config: AdjointMatchingConfig = AdjointMatchingConfig()


class DiffusionModelAdapter(nn.Module):
    """
    CRITICAL ADAPTER: Bridges the gap between Adjoint Matching (DDPM-style)
    and Elucidated Diffusion (EDM/Karras-style).
    
    1. Translates Solver Time (t_idx) -> Adjoint Matching Sigma -> Model Input.
    2. Translates Model Output (Denoised X) -> Noise Prediction (Epsilon).
    """
    def __init__(self, elucidated_model: ElucidatedDiffusion, solver_config: AdjointMatchingConfig):
        super().__init__()
        self.model = elucidated_model
        self.config = solver_config
        
        # We need the EXACT sigma schedule used by the solver to ensure consistency.
        # sigma(t) = sqrt( 2(1 - t + h) / (t + h) ) [Paper Eq 235]
        self.h = 1.0 / self.config.num_inference_steps

    def get_sigma_at_step(self, t_idx_tensor):
        """
        Calculates sigma for a given integer timestep index k.
        t_continuous = k * h
        """
        # t_idx is roughly [0, 1000]. We map to [0, 1] based on K steps.
        # The solver passes indices relevant to num_inference_steps (e.g. 0..40)
        # IF the solver passed raw indices 0..1000, we map differently.
        # Looking at AdjointMatchingSolver: it uses `timesteps = linspace(1000, 0, 40)`.
        
        # Map integer index [0, 1000] -> continuous t [0, 1]
        t_continuous = t_idx_tensor.float() / self.config.num_train_timesteps
        
        numerator = 2 * (1 - t_continuous + self.h)
        denominator = t_continuous + self.h
        sigma = torch.sqrt(numerator / denominator)
        return sigma

    def forward(self, x, t_idx, prompt_emb=None):
        """
        Args:
            x: Noisy input (Batch, Dim)
            t_idx: Timestep indices (Batch,)
        Returns:
            pred_epsilon: Estimated noise (Batch, Dim)
        """
        # 1. Calculate the sigma mandated by Adjoint Matching physics
        sigmas = self.get_sigma_at_step(t_idx)
        
        # 2. Reshape for broadcasting
        # ElucidatedDiffusion expects sigmas shape (Batch,) usually, 
        # but internal network might expect (Batch, 1)
        sigmas = sigmas.view(x.shape[0])
        
        # 3. Forward pass through Pre-trained EDM Model
        # ElucidatedDiffusion.preconditioned_network_forward outputs D(x, sigma) (Denoised data)
        # Note: We must ensure we don't accidentally add more noise or compute loss.
        # We use the underlying preconditioned call.
        denoised_x = self.model.preconditioned_network_forward(x, sigmas, clamp=False)
        
        # 4. Convert Denoised Data -> Epsilon
        # x = data + sigma * epsilon  =>  epsilon = (x - data) / sigma
        
        # Reshape sigma for broadcasting against x
        sigmas_broad = sigmas.view(-1, *([1] * (x.ndim - 1)))
        
        pred_epsilon = (x - denoised_x) / (sigmas_broad + 1e-5) # Stability epsilon
        
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
    
    args = parser.parse_args()

    # 1. Setup
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # 2. Data Loading (For shape and normalization context)
    # Even though S-MEME generates data from noise, we need the dataset 
    # to construct the model architecture correctly (dim size) and normalization stats.
    print(f"Loading dataset info for {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    rew = d4rl_dataset['rewards'][:, None] if len(d4rl_dataset['rewards'].shape) == 1 else d4rl_dataset['rewards']
    term = d4rl_dataset['terminals'][:, None] if len(d4rl_dataset['terminals'].shape) == 1 else d4rl_dataset['terminals']
    next_obs = d4rl_dataset['next_observations']
    
    # Concatenate to determine input dimension
    inputs_np = np.concatenate([obs, act, rew, next_obs, term], axis=1)
    # We don't necessarily need a TensorDataset for training if we generate noise on the fly,
    # but the Solver expects a DataLoader-like structure for the inner loop.
    
    # Creates a dummy loader that just yields noise (or real data if we wanted support constraints)
    # For Pure S-MEME, we just need the shape.
    input_dim = inputs_np.shape[1]
    print(f"Data Dimension: {input_dim}")

    # 3. Model Construction & Loading
    # Construct base model structure
    dummy_inputs = torch.zeros(1, input_dim)
    base_model_edm = construct_diffusion_model(inputs=dummy_inputs)
    
    print(f"Loading weights from {args.load_checkpoint}...")
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # Load weights
    base_model_edm.load_state_dict(state_dict)
    base_model_edm.to(device)
    base_model_edm.eval() # Freeze stats

    # 4. Wrap the Model for Adjoint Matching
    # This converts the EDM (Denoising) model into a DDPM (Epsilon) interface
    # compatible with our solver.
    
    am_config = AdjointMatchingConfig()
    wrapped_model = DiffusionModelAdapter(base_model_edm, am_config).to(device)
    
    # 5. Configure S-MEME
    smeme_config = SMEMEConfig(
        num_smeme_iterations=args.iterations,
        alpha_schedule=tuple(args.alphas),
        am_config=am_config
    )
    
    # 6. Initialize Solver
    # SMEMESolver expects the wrapped model (epsilon predictor)
    solver = SMEMESolver(base_model=wrapped_model, config=smeme_config)
    
    # 7. Training Loop
    # We construct a generator that yields random noise batches
    def noise_generator(batch_size, dim):
        while True:
            yield torch.randn(batch_size, dim).to(device)
            
    # We essentially manually run the solver's internal loop control
    # relying on the SMEMESolver structure provided previously.
    
    # HACK: The SMEMESolver.train expects a loader. 
    # We create a finite iterator for the number of steps requested.
    class InfiniteNoiseLoader:
        def __init__(self, batch_size, dim, limit):
            self.batch_size = batch_size
            self.dim = dim
            self.limit = limit
            self.cnt = 0
        def __iter__(self):
            return self
        def __next__(self):
            if self.cnt >= self.limit:
                raise StopIteration
            self.cnt += 1
            return torch.randn(self.batch_size, self.dim).to(device)
            
    train_loader = InfiniteNoiseLoader(args.batch_size, input_dim, args.steps_per_iter)
    
    print("Starting S-MEME Training...")
    
    # Run S-MEME
    # This returns the fine-tuned Wrapped Model
    finetuned_wrapper = solver.train(train_loader)
    
    # 8. Save Results
    # We need to extract the underlying EDM model state dict to stay compatible with Synther
    finetuned_edm = finetuned_wrapper.current_model.model # Unwrap Adapter -> ElucidatedDiffusion
    
    save_path = results_folder / "smeme_finetuned.pt"
    torch.save({
        'model': finetuned_edm.state_dict(),
        'config': smeme_config, 
        'steps': args.steps_per_iter * args.iterations
    }, save_path)
    
    print(f"Model saved to {save_path}")
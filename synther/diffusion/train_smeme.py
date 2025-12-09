# synther/diffusion/train_smeme.py

import argparse
import copy
import pathlib
import gin
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from einops import reduce

# Import existing infrastructure to ensure compatibility
from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_diffuser import get_gymnasium_id
from just_d4rl import d4rl_offline_dataset

@gin.configurable
class SMEMETrainer(Trainer):
    """
    State-of-the-art S-MEME Trainer adapted for Elucidated Diffusion (Karras et al.).
    
    Implements the update rule:
        Target_Score = (1 - lambda) * Reference_Score
    
    In terms of the denoiser output D(x):
        D_target(x) = x + (1 - lambda) * (D_ref(x) - x)
    """
    def __init__(
        self,
        diffusion_model,
        ref_model,
        smeme_step_size: float = 0.1, # The 'lambda' parameter
        **kwargs
    ):
        super().__init__(diffusion_model, **kwargs)
        self.smeme_step_size = smeme_step_size
        
        # Prepare Reference Model
        # We assume ref_model is already loaded with weights
        self.ref_model = ref_model
        self.ref_model.eval()
        self.ref_model.requires_grad_(False)
        
        # Move ref_model to correct device immediately if possible, 
        # though accelerator handles the main model.
        # We will handle ref_model placement dynamically in the loop or here.
        self.ref_model.to(self.accelerator.device)

    def smeme_loss(self, inputs):
        """
        Computes the S-MEME fine-tuning loss.
        """
        # 1. Standard Elucidated Diffusion Setup
        inputs = self.model.normalizer.normalize(inputs)
        batch_size = inputs.shape[0]
        
        # Sample noise levels (sigmas) - Same log-normal dist as training
        sigmas = self.model.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.model.event_shape)))
        
        # Add noise
        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise
        
        # 2. Compute Reference Output (Gradient-Free)
        with torch.no_grad():
            # Ensure ref_model is on the same device as inputs
            # (Accelerate might have moved inputs to a specific GPU)
            if self.ref_model.device != inputs.device:
                self.ref_model.to(inputs.device)
                
            # Get D_ref(x_t)
            # We use preconditioned_network_forward to get the raw denoiser output
            denoised_ref = self.ref_model.preconditioned_network_forward(
                noised_inputs, sigmas, clamp=False
            )
            
        # 3. Compute S-MEME Target
        # Formula: D_target = x_t + (1 - lambda) * (D_ref - x_t)
        # This dampens the score magnitude by lambda, increasing entropy.
        
        # (D_ref - x_t) is the "vector pointing to data".
        # We scale this vector down.
        score_direction = denoised_ref - noised_inputs
        target = noised_inputs + (1.0 - self.smeme_step_size) * score_direction
        
        # 4. Compute Current Model Output
        # We need to access the underlying model if it's wrapped by DDP/Accelerator
        # But calling self.model(...) calls forward() which computes loss.
        # We need to call the denoiser forward pass directly.
        
        # Unwrapping is tricky in training loop, but we can call the method directly 
        # if the model exposes it. ElucidatedDiffusion does.
        # If wrapped by DDP, methods are accessible via attributes or direct call if properly configured.
        # However, safely, we can just call the method on self.model because DDP forwards calls.
        
        denoised_pred = self.model.preconditioned_network_forward(
            noised_inputs, sigmas, clamp=False
        )
        
        # 5. Compute Weighted Loss
        # Karras weighting scheme: (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
        weights = self.model.loss_weight(sigmas)
        
        losses = F.mse_loss(denoised_pred, target, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * weights
        
        return losses.mean()

    def train(self):
        """
        Custom training loop overriding the standard Trainer.train()
        to use smeme_loss instead of standard diffusion loss.
        """
        accelerator = self.accelerator
        device = accelerator.device

        print(f"Starting S-MEME Fine-tuning for {self.train_num_steps} steps...")
        
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    # Load batch from dataset (Manifold Support)
                    data = (next(self.dl)[0]).to(device)

                    with self.accelerator.autocast():
                        # --- CRITICAL CHANGE: Use S-MEME Loss ---
                        loss = self.smeme_loss(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Logging
                pbar.set_description(f'S-MEME loss: {total_loss:.4f}')
                wandb.log({
                    'step': self.step,
                    'loss': total_loss,
                    'lr': self.opt.param_groups[0]['lr']
                })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                
                # EMA Update
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    # Save checkpoints
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.save(self.step)

                pbar.update(1)

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        accelerator.print('S-MEME Fine-tuning complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="D4RL dataset name (e.g., hopper-medium-replay-v2)")
    parser.add_argument('--load_checkpoint', type=str, required=True, help="Path to the pre-trained model (e.g., results/model-100000.pt)")
    parser.add_argument('--results_folder', type=str, default='./results_smeme', help="Folder to save fine-tuned model")
    
    # S-MEME Parameters
    parser.add_argument('--smeme_step_size', type=float, default=0.1, help="Entropy maximization strength (lambda). 0.1 is standard.")
    parser.add_argument('--train_num_steps', type=int, default=10000, help="Number of fine-tuning steps")
    
    # Standard Config
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    parser.add_argument('--wandb-project', type=str, default="smeme-finetuning")
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()

    # 1. Parse Config
    # We must append the pre-trained checkpoint config to ensure architecture match
    # However, usually we just use the same .gin file.
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # 2. Set Seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 3. Load Data (Required for Manifold Support)
    # We need to reconstruct the dataset to fine-tune on the valid manifold
    from just_d4rl import d4rl_offline_dataset
    print(f"Loading dataset {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    next_obs = d4rl_dataset['next_observations']
    rew = d4rl_dataset['rewards']
    terminals = d4rl_dataset['terminals']

    if len(rew.shape) == 1: rew = rew[:, None]
    if len(terminals.shape) == 1: terminals = terminals[:, None]

    # Concatenate to (s, a, r, s', t)
    inputs_np = np.concatenate([obs, act, rew, next_obs, terminals], axis=1)
    inputs = torch.from_numpy(inputs_np).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    # 4. Construct Models
    print("Constructing models...")
    
    # A. Trainable Model
    model = construct_diffusion_model(inputs=inputs)
    
    # B. Reference Model (Identical Architecture)
    ref_model = construct_diffusion_model(inputs=inputs)

    # 5. Load Weights
    print(f"Loading weights from {args.load_checkpoint}...")
    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
    
    # Handle state_dict structure
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict)
    
    # 6. Initialize S-MEME Trainer
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Save config for reproducibility
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    trainer = SMEMETrainer(
        diffusion_model=model,
        ref_model=ref_model,
        dataset=dataset,
        results_folder=args.results_folder,
        train_num_steps=args.train_num_steps,
        smeme_step_size=args.smeme_step_size,
        # Default params from gin or defaults
    )
    
    # Override the EMA model with the pre-trained weights as well
    # (Trainer init creates a new EMA, so we must load it)
    if 'ema' in checkpoint:
        print("Loading EMA weights...")
        trainer.ema.load_state_dict(checkpoint['ema'])
    else:
        print("Warning: No EMA weights found in checkpoint. EMA started from scratch (or matched model).")

    # 7. Start Fine-Tuning
    wandb.init(project=args.wandb_project, config=args)
    trainer.train()
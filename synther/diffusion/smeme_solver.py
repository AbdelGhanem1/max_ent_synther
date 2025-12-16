import torch
import copy
import numpy as np
from tqdm import tqdm
import wandb
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver

# Use torch.amp for PyTorch 2.x compatibility
from torch.amp import autocast, GradScaler 

class SMEMESolver:
    def __init__(self, base_model, config):
        """
        Implements S-MEME (Algorithm 1) using Adjoint Matching as the linear solver.
        """
        self.current_model = base_model
        self.config = config
        
        # FIX 1: Explicitly force the current model to be trainable
        self.current_model.train()
        for p in self.current_model.parameters():
            p.requires_grad = True
        
        # Create the reference model (Previous Iteration)
        self.previous_model = copy.deepcopy(base_model)
        self.previous_model.eval()
        self.previous_model.requires_grad_(False)

    def _get_score_at_data(self, model, x_data, t_idx):
        """
        Computes the TRUE Score s(x, t) = -epsilon / sigma.
        Now includes the critical 1/sigma scaling to prevent vanishing gradients.
        """
        # Create timestep tensor [Batch,]
        t_tensor = torch.tensor([t_idx], device=x_data.device).repeat(x_data.shape[0])
        
        with torch.no_grad():
            with autocast(device_type='cuda'):
                # 1. Get Noise Prediction (Epsilon)
                eps = model(x_data, t_tensor)
                
                # 2. Get Sigma (Noise Level) from the Adapter
                # The model passed here is DiffusionModelAdapter, so it has this method.
                if hasattr(model, 'get_sigma_at_step'):
                    sigma = model.get_sigma_at_step(t_tensor)
                else:
                    # Fallback if unwrapped (unlikely given your train script)
                    # Assume t_idx maps linearly 0->1 if we can't ask the model
                    sigma = torch.tensor(1.0, device=x_data.device) 
                    print("⚠️ Warning: Model has no get_sigma_at_step. Using sigma=1.0")

                # 3. Reshape for broadcasting [Batch, 1]
                sigma = sigma.view(-1, 1)
                
                # 4. Compute Raw Score (The Nuclear Option)
                # We add a tiny clamp (1e-5) just to prevent literal NaN division
                # but allow it to get MASSIVE (e.g., 10,000).
                safe_sigma = torch.maximum(sigma, torch.tensor(1e-5, device=x_data.device))
                score = -eps / safe_sigma 
            
        return score

    def train(self, train_loader):
        print(f"Starting S-MEME with K={self.config.num_smeme_iterations} iterations.")
        
        # Initialize Scaler for Mixed Precision
        scaler = GradScaler('cuda')
        
        total_steps = 0
        
        for k in range(self.config.num_smeme_iterations):
            print(f"\n=== S-MEME Iteration {k+1}/{self.config.num_smeme_iterations} ===")
            
            # Ensure model is in training mode for this iteration
            self.current_model.train()
            self.current_model.requires_grad_(True)
            
            alpha = self.config.alpha_schedule[k]
            self.config.am_config.reward_multiplier = 1.0 / alpha
            
            print(f"   > Alpha: {alpha} (Reward Multiplier: {self.config.am_config.reward_multiplier:.2f})")
            
            solver = VectorFieldAdjointSolver(
                model_pre=self.previous_model,
                model_fine=self.current_model,
                config=self.config.am_config
            )
            
            def entropy_gradient_fn(x, t_idx):
                score = self._get_score_at_data(self.previous_model, x, t_idx)
                return score 
            
            pbar = tqdm(train_loader, desc=f"Iter {k+1}", dynamic_ncols=True)
            running_loss = 0.0
            
            for step, batch in enumerate(pbar):
                self.current_model.zero_grad()
                
                # --- OPTIMIZATION: Mixed Precision Context ---
                with autocast(device_type='cuda'):
                    loss = solver.solve_vector_field(batch, entropy_gradient_fn)
                
                # --- FIX 3: Robust Gradient Check ---
                if torch.isnan(loss):
                    # Warning for numerical explosion
                    # Only print occasionally to avoid spamming
                    if step % 10 == 0:
                        print(f"⚠️ Warning: Loss is NaN at step {step}. Skipping.")
                    continue

                if loss.grad_fn is None:
                    # Warning for broken graph (prevents crash)
                    # This usually implies active_indices was empty or model was frozen
                    if step % 100 == 0: 
                        print(f"⚠️ Warning: Loss has no gradient at step {step}. Skipping.")
                    continue
                
                # --- OPTIMIZATION: Scaled Backward Pass ---
                scaler.scale(loss).backward()
                
                # Create optimizer if not exists
                if not hasattr(self, 'optimizer'):
                    self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=1e-4, weight_decay=1e-2)
                
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                
                scaler.step(self.optimizer)
                scaler.update()
                
                # Logging
                loss_val = loss.item()
                running_loss = 0.9 * running_loss + 0.1 * loss_val if step > 0 else loss_val
                
                pbar.set_postfix({'loss': f"{running_loss:.4f}"})
                
                if wandb.run is not None:
                    wandb.log({
                        "smeme/loss": loss_val,
                        "smeme/iteration": k + 1,
                        "smeme/alpha": alpha,
                        "total_steps": total_steps
                    })
                
                total_steps += 1
                
            # Update Reference Model
            self.previous_model.load_state_dict(self.current_model.state_dict())
            self.previous_model.eval()
            self.previous_model.requires_grad_(False)
            
        return self.current_model


class VectorFieldAdjointSolver(AdjointMatchingSolver):
    """
    Subclass of AdjointMatchingSolver that overrides the initialization step.
    """
    
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        """
        Modified solve method for S-MEME with AMP support.
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1, 0, 
            self.config.num_inference_steps, 
            device=device, dtype=torch.long
        )
        
        # 1. Forward Pass
        traj = [x_0]
        curr_x = x_0
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.tensor([t], device=device).repeat(batch_size)
                noise_pred = self.model_fine(curr_x, t_tensor, prompt_emb)
                prev_x, _, _ = self.scheduler.step(
                    noise_pred, t.item(), curr_x, 
                    eta=self.config.eta, num_inference_steps=self.config.num_inference_steps
                )
                traj.append(prev_x)
                curr_x = prev_x

        # 2. Adjoint Initialization
        x_prev = traj[-2]
        t_last = timesteps[-1].item()
        
        with torch.no_grad():
            t_tensor = torch.tensor([t_last], device=device).repeat(batch_size)
            noise_pred = self.model_fine(x_prev, t_tensor, prompt_emb)
            x_final_clean, _, _ = self.scheduler.step(
                noise_pred, t_last, x_prev, eta=0.0,
                num_inference_steps=self.config.num_inference_steps
            )
            
            reward_grad = vector_field_fn(x_final_clean, 0)
            
        adjoint_state = -self.config.reward_multiplier * reward_grad
        
        # 3. Backward Pass
        adjoint_storage = {}
        for k in range(self.config.num_inference_steps - 1, -1, -1):
            t_val = timesteps[k].item()
            x_k = traj[k]
            vjp = self._grad_inner_product(x_k, t_val, adjoint_state, prompt_emb)
            adjoint_state = adjoint_state + vjp
            adjoint_storage[k] = adjoint_state
            
        # 4. Loss Computation
        active_indices = self._sample_time_indices(
            self.config.num_inference_steps, 
            self.config.num_timesteps_to_load, 
            device
        )
        
        # FIX 2: Initialize as Python float to accept gradients naturally
        total_loss = 0.0 
        
        # Debug check for empty indices (should not happen with default configs)
        if len(active_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        for k in active_indices:
            k = k.item()
            t_val = timesteps[k].item()
            x_k = traj[k].detach()
            target_adjoint = adjoint_storage[k].detach()
            
            t_tensor = torch.tensor([t_val], device=device).repeat(batch_size)
            eps_fine = self.model_fine(x_k, t_tensor, prompt_emb)
            with torch.no_grad():
                eps_pre = self.model_pre(x_k, t_tensor, prompt_emb)
                
            _, _, std_dev_t = self.scheduler.step(eps_fine, t_val, x_k, eta=self.config.eta, 
                                                  num_inference_steps=self.config.num_inference_steps)
            
            diff = (eps_fine - eps_pre)
            target = target_adjoint * std_dev_t.view(-1, 1)
            loss_per_sample = torch.sum((diff + target) ** 2, dim=1)
            
            # EMA Clipping
            loss_root = torch.sqrt(loss_per_sample.detach())
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            mask = (loss_root < ema_threshold).float()
            
            total_loss = total_loss + (loss_per_sample * mask).mean()
            
        return total_loss
import torch
import copy
import numpy as np
from tqdm import tqdm
import wandb
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from torch.amp import autocast, GradScaler

class SMEMESolver:
    def __init__(self, base_model, config):
        """
        Implements S-MEME (Algorithm 1) using Adjoint Matching as the linear solver.
        """
        self.current_model = base_model
        self.config = config
        
        self.previous_model = copy.deepcopy(base_model)
        self.previous_model.eval()
        self.previous_model.requires_grad_(False)

    def _get_score_at_data(self, model, x_data, t_idx):
        """
        Computes the Score using the CORRECT EDM Physics.
        Score = -epsilon / sigma
        
        [MODIFIED] Forces Float32 to prevent FP16 Overflow when alpha is small.
        """
        # 1. Create Time Tensor
        t_tensor = torch.tensor([t_idx], device=x_data.device).repeat(x_data.shape[0])
        
        # [CRITICAL FIX] Disable autocast for the score division.
        # This ensures the division and large values stay in Float32.
        with torch.no_grad():
             # Run model in whatever precision it wants, but cast output to float32
            with autocast(device_type='cuda', enabled=False):
                # If model expects half, we might need a wrapper, but usually models handle float32 input fine.
                # To be safe, we let the model run in autocast if needed, but we do the DIVISION in float32.
                pass

            # 2. Get Epsilon (Noise Prediction)
            # We run forward pass. If inside autocast context in train loop, this might be FP16.
            # We cast result to float32 immediately.
            eps = model(x_data, t_tensor).float()
            
            # 3. GET THE CORRECT SIGMA
            if hasattr(model, 'get_sigma_at_step'):
                sigma = model.get_sigma_at_step(t_tensor)
            else:
                sigma = torch.tensor(1.0, device=x_data.device)

            # Reshape for broadcasting
            sigma = sigma.view(-1, 1).float()
            
            # 4. Compute Score with Safety Clamp
            # Limit sigma to >= 0.05 to prevent explosion
            safe_sigma = torch.maximum(sigma, torch.tensor(0.05, device=x_data.device))
            
            score = -eps / safe_sigma
            
            # [SAFETY CLAMP] Cap the score magnitude to prevent infinite gradients
            # A score of 100 corresponds to a massive force. 
            score = torch.clamp(score, min=-100.0, max=100.0)
            
        return score

    def train(self, train_loader):
        print(f"Starting S-MEME with K={self.config.num_smeme_iterations} iterations.")
        
        total_steps = 0
        
        for k in range(self.config.num_smeme_iterations):
            print(f"\n=== S-MEME Iteration {k+1}/{self.config.num_smeme_iterations} ===")
            
            alpha = self.config.alpha_schedule[k]
            self.config.am_config.reward_multiplier = 1.0 / alpha
            
            print(f"   > Alpha: {alpha} (Reward Multiplier: {self.config.am_config.reward_multiplier:.2f})")
            
            solver = VectorFieldAdjointSolver(
                model_pre=self.previous_model,
                model_fine=self.current_model,
                config=self.config.am_config
            )
            
            def entropy_gradient_fn(x, t_idx):
                # Returns Score (which points Uphill). 
                # Solver applies negative sign to move Downhill (Entropy).
                score = self._get_score_at_data(self.previous_model, x, t_idx)
                return score 
            
            pbar = tqdm(train_loader, desc=f"Iter {k+1}", dynamic_ncols=True)
            running_loss = 0.0
            
            # Initialize optimizer if needed
            if not hasattr(self, 'optimizer'):
                 self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=1e-4, weight_decay=1e-2)

            for step, batch in enumerate(pbar):
                self.current_model.zero_grad()
                
                # [MODIFIED] Ensure solver output is float32
                loss = solver.solve_vector_field(batch, entropy_gradient_fn)
                
                # Check for NaN immediately
                if torch.isnan(loss):
                    print(f"\n[WARNING] Loss is NaN at step {step}! Skipping step.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                
                self.optimizer.step()
                
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
                
            self.previous_model.load_state_dict(self.current_model.state_dict())
            self.previous_model.eval()
            self.previous_model.requires_grad_(False)
            
        return self.current_model


class VectorFieldAdjointSolver(AdjointMatchingSolver):
    
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
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
            
            # [CRITICAL] Reward grad might be large, keep float32
            reward_grad = vector_field_fn(x_final_clean, 0).float()
            
            # [STABILITY FIX] Normalize the reward gradient
            # The raw score can be huge (100+). We normalize it to have a reasonable scale.
            # This prevents the initial "kick" of the backward ODE from being Infinite.
            grad_norm = torch.norm(reward_grad.reshape(reward_grad.shape[0], -1), dim=1, keepdim=True)
            # Clip the norm to a max value (e.g., 1.0 or 10.0) to prevent explosion
            # but allow small gradients to pass through.
            scale_factor = torch.clamp(grad_norm, min=1.0) 
            reward_grad = reward_grad / scale_factor
            
        adjoint_state = -self.config.reward_multiplier * reward_grad
        
        # 3. Backward Pass
        adjoint_storage = {}
        for k in range(self.config.num_inference_steps - 1, -1, -1):
            t_val = timesteps[k].item()
            x_k = traj[k]
            vjp = self._grad_inner_product(x_k, t_val, adjoint_state, prompt_emb)
            adjoint_state = adjoint_state + vjp
            adjoint_storage[k] = adjoint_state
            
        # 4. Loss Computation (Float32 forced)
        active_indices = self._sample_time_indices(
            self.config.num_inference_steps, 
            self.config.num_timesteps_to_load, 
            device
        )
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for k in active_indices:
            k = k.item()
            t_val = timesteps[k].item()
            x_k = traj[k].detach()
            target_adjoint = adjoint_storage[k].detach()
            
            t_tensor = torch.tensor([t_val], device=device).repeat(batch_size)
            
            # Force Float32 for model output and loss
            eps_fine = self.model_fine(x_k, t_tensor, prompt_emb).float()
            
            with torch.no_grad():
                eps_pre = self.model_pre(x_k, t_tensor, prompt_emb).float()
                
            _, _, std_dev_t = self.scheduler.step(eps_fine, t_val, x_k, eta=self.config.eta, 
                                                  num_inference_steps=self.config.num_inference_steps)
            
            diff = (eps_fine - eps_pre)
            
            # [FIX] Force target calculation in float32
            target = target_adjoint.float() * std_dev_t.view(-1, 1).float()
            
            # Loss Calculation
            loss_per_sample = torch.sum((diff + target) ** 2, dim=1)
            
            # EMA Clipping (Ensure float32)
            loss_root = torch.sqrt(loss_per_sample.detach())
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            mask = (loss_root < ema_threshold).float()
            
            total_loss += (loss_per_sample * mask).mean()
            
        return total_loss
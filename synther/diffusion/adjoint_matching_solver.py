import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

class DDIMScheduler:
    """
    Minimal implementation of SOCDDIMScheduler logic for general vector data.
    Manages alphas, betas, and the update step (Eq 12 in DDIM paper).
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0, device=device) # For t < 0

    def step(self, 
             model_output: torch.Tensor, 
             timestep: int, 
             sample: torch.Tensor, 
             eta: float = 0.0,
             num_inference_steps: int = 40) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one backward step (generation) from t to t-1.
        Matches 'SOCDDIMScheduler.step'.
        
        Returns:
            prev_sample (x_{t-1})
            pred_original_sample (x_0 prediction)
            std_dev_t (sigma_t for this step)
        """
        # 1. Calculate step indices
        step_ratio = self.num_train_timesteps // num_inference_steps
        prev_timestep = timestep - step_ratio

        # 2. Get alpha_bar terms
        alpha_prod_t = self.alphas_cumprod[timestep]
        
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        
        # 3. Compute predicted x_0 (pred_original_sample)
        pred_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        # 4. Compute variance (sigma_t)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * (variance ** 0.5)

        # 5. Compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2)**0.5 * model_output

        # 6. Compute prev_sample (x_{t-1}) deterministic part
        prev_sample = (alpha_prod_t_prev ** 0.5) * pred_original_sample + pred_sample_direction

        # 7. Add noise (if eta > 0)
        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = prev_sample + std_dev_t * noise
            
        return prev_sample, pred_original_sample, std_dev_t


class AdjointMatchingSolver:
    def __init__(self, 
                 model_pre: nn.Module, 
                 model_fine: nn.Module, 
                 config):
        """
        Args:
            model_pre: Frozen pre-trained noise predictor.
            model_fine: Trainable noise predictor.
            config: Configuration object with keys:
                - num_inference_steps (K): e.g. 40
                - num_train_timesteps: e.g. 1000
                - num_timesteps_to_load: Batch size for loss computation
                - per_sample_threshold_quantile: e.g. 0.9 (for EMA clipping)
                - reward_multiplier: Scale for reward gradient
        """
        self.model_pre = model_pre
        self.model_fine = model_fine
        self.config = config
        
        # Scheduler setup
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            device=next(model_fine.parameters()).device
        )
        
        # EMA Tracking for Loss Clipping
        self.ema_value = -1
        self.ema_updates = 0
        self.ema_decay = 0.9

    def _ema_update(self, value):
        """Updates the Exponential Moving Average of the loss quantile."""
        if self.ema_updates == 0:
            self.ema_value = value
        elif self.ema_updates < 1.0 / self.ema_decay:
            # Warmup
            self.ema_value = value / (self.ema_updates + 1) + \
                             (1 - 1 / (self.ema_updates + 1)) * self.ema_value
        else:
            # Standard EMA
            self.ema_value = self.ema_decay * self.ema_value + (1 - self.ema_decay) * value
        
        self.ema_updates += 1
        return self.ema_value

    def _sample_time_indices(self, num_timesteps, num_to_load, device):
        """
        Stratified Sampling. Handles requests > population size.
        """
        if num_to_load >= num_timesteps:
            return torch.arange(num_timesteps, device=device, dtype=torch.long)

        middle_timestep = round(num_timesteps * 0.6)
        
        pop1 = np.arange(0, middle_timestep)
        pop2 = np.arange(middle_timestep, num_timesteps)
        
        goal_1 = num_to_load // 2
        goal_2 = num_to_load - goal_1
        
        count_2 = min(len(pop2), goal_2)
        remainder = goal_2 - count_2
        count_1 = min(len(pop1), goal_1 + remainder)
        
        indices_1 = np.random.choice(pop1, count_1, replace=False)
        indices_2 = np.random.choice(pop2, count_2, replace=False)
        
        indices = np.concatenate((indices_1, indices_2))
        return torch.tensor(np.sort(indices), dtype=torch.long, device=device)

    def _grad_inner_product(self, x, t_idx, vector, prompt_emb=None):
        """Computes VJP (Vector-Jacobian Product) for the Adjoint ODE."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            
            # Predict noise using FROZEN model_pre
            t_tensor = torch.tensor([t_idx], device=x.device).repeat(x.shape[0])
            noise_pred = self.model_pre(x, t_tensor, prompt_emb) 
            
            # Compute "Next Sample"
            prev_sample, _, _ = self.scheduler.step(
                noise_pred, t_idx, x, eta=0.0, 
                num_inference_steps=self.config.num_inference_steps
            )
            
            drift = prev_sample - x
            sum_inner_prod = torch.sum(drift * vector)
            
            grad = torch.autograd.grad(sum_inner_prod, x)[0]
            
        return grad.detach()

    def solve(self, x_0, reward_fn, prompt_emb=None):
        """
        Main execution of Algorithm 2 (Linear Solver).
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1, 0, 
            self.config.num_inference_steps, 
            device=device, dtype=torch.long
        )
        
        # --- 1. Forward Pass (Sampling) ---
        traj = [x_0]
        curr_x = x_0
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.tensor([t], device=device).repeat(batch_size)
                noise_pred = self.model_fine(curr_x, t_tensor, prompt_emb)
                prev_x, _, _ = self.scheduler.step(
                    noise_pred, t.item(), curr_x, 
                    eta=self.config.eta, # Usually 1.0 for training exploration
                    num_inference_steps=self.config.num_inference_steps
                )
                traj.append(prev_x)
                curr_x = prev_x
                
        # --- 2. Adjoint Initialization ---
        # "Noiseless update" trick from Paper Appendix H.1.
        
        x_prev = traj[-2] 
        t_last = timesteps[-1].item() # t=0
        
        with torch.no_grad():
            # Use model_fine for this final clean prediction
            t_tensor = torch.tensor([t_last], device=device).repeat(batch_size)
            noise_pred = self.model_fine(x_prev, t_tensor, prompt_emb)
            
            # Deterministic step (eta=0)
            x_final_clean, _, _ = self.scheduler.step(
                noise_pred, t_last, x_prev, eta=0.0,
                num_inference_steps=self.config.num_inference_steps
            )
        
        # FIX: Detach to make x_final_clean a leaf node.
        # We need the gradient of the Reward function at this point.
        # We do NOT need to backprop through the denoising step itself.
        x_final_clean = x_final_clean.detach().requires_grad_(True)
        
        r = reward_fn(x_final_clean).sum()
        r.backward()
        reward_grad = x_final_clean.grad.detach() # Now this will work
            
        # Initialize Adjoint State
        adjoint_state = -self.config.reward_multiplier * reward_grad
        
        adjoint_storage = {} 
        
        # --- 3. Backward Pass (Lean Adjoint ODE) ---
        for k in range(self.config.num_inference_steps - 1, -1, -1):
            t_val = timesteps[k].item()
            x_k = traj[k]
            
            vjp = self._grad_inner_product(x_k, t_val, adjoint_state, prompt_emb)
            adjoint_state = adjoint_state + vjp
            adjoint_storage[k] = adjoint_state
            
        # --- 4. Loss Computation ---
        active_indices = self._sample_time_indices(
            self.config.num_inference_steps, 
            self.config.num_timesteps_to_load, 
            device
        )
        
        total_loss = 0.0
        
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
            target = target_adjoint * std_dev_t.view(-1, 1) # broadcasting
            
            loss_per_sample = torch.sum((diff + target) ** 2, dim=1)
            
            # --- 5. EMA Quantile Clipping ---
            loss_root = torch.sqrt(loss_per_sample.detach())
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            mask = (loss_root < ema_threshold).float()
            
            loss_final = (loss_per_sample * mask).mean()
            total_loss += loss_final
            
        return total_loss
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
        # In generation, timestep is the high value (e.g. 975), prev is lower (e.g. 950)
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
        # Assuming prediction_type == "epsilon" (standard)
        # x_0 = (x_t - sqrt(1-alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        pred_original_sample = (sample - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)

        # 4. Compute variance (sigma_t)
        # Formula (16) in DDIM paper
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
        
        # EMA Tracking for Loss Clipping (from am_trainer.py)
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
        Stratified Sampling (am_trainer.py: sample_time_indices).
        Splits at 60% of process. Samples half from early (noisy), half from late (structure).
        
        FIX: Handles cases where requested sample size exceeds population size.
        """
        # If we want to load everything (or more), just return all indices
        if num_to_load >= num_timesteps:
            return torch.arange(num_timesteps, device=device, dtype=torch.long)

        middle_timestep = round(num_timesteps * 0.6)
        
        # Define populations
        pop1 = np.arange(0, middle_timestep)
        pop2 = np.arange(middle_timestep, num_timesteps)
        
        # Define goals (try to split 50/50)
        goal_1 = num_to_load // 2
        goal_2 = num_to_load - goal_1
        
        # Adjust goals if populations are too small
        # If pop2 is too small for goal_2, take all of pop2 and add remainder to goal_1
        count_2 = min(len(pop2), goal_2)
        remainder = goal_2 - count_2
        
        count_1 = min(len(pop1), goal_1 + remainder)
        
        # Sampling
        indices_1 = np.random.choice(pop1, count_1, replace=False)
        indices_2 = np.random.choice(pop2, count_2, replace=False)
        
        indices = np.concatenate((indices_1, indices_2))
        return torch.tensor(np.sort(indices), dtype=torch.long, device=device)

    def _grad_inner_product(self, x, t_idx, vector, prompt_emb=None):
        """
        Computes VJP (Vector-Jacobian Product) for the Adjoint ODE.
        Corresponds to `grad_inner_product` in am_trainer.py.
        
        Calculates: grad_x( (drift_pre(x) * adjoint_vector).sum() )
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            
            # Predict noise using FROZEN model_pre (Base Dynamics)
            # t_idx is the integer timestep (e.g., 975)
            t_tensor = torch.tensor([t_idx], device=x.device).repeat(x.shape[0])
            noise_pred = self.model_pre(x, t_tensor, prompt_emb) 
            
            # Compute "Next Sample" using scheduler logic
            # We only need the deterministic drift part: prev_sample - x
            prev_sample, _, _ = self.scheduler.step(
                noise_pred, t_idx, x, eta=0.0, 
                num_inference_steps=self.config.num_inference_steps
            )
            
            # drift = prev_sample - x
            # inner_prod = sum(drift * vector)
            drift = prev_sample - x
            sum_inner_prod = torch.sum(drift * vector)
            
            # Compute Gradient w.r.t x
            grad = torch.autograd.grad(sum_inner_prod, x)[0]
            
        return grad.detach()

    def solve(self, x_0, reward_fn, prompt_emb=None):
        """
        Main execution of Algorithm 2 (Linear Solver).
        
        Args:
            x_0: Initial noise batch.
            reward_fn: Function mapping x -> scalar reward (entropy first variation).
            prompt_emb: Optional conditioning embedding.
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        
        # Discrete timesteps for generation (e.g. 975, 950, ..., 0)
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1, 0, 
            self.config.num_inference_steps, 
            device=device, dtype=torch.long
        )
        
        # --- 1. Forward Pass (Sampling) ---
        # Generate trajectory using CURRENT POLICY (model_fine)
        traj = [x_0]
        curr_x = x_0
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.tensor([t], device=device).repeat(batch_size)
                
                # Model Fine Prediction
                noise_pred = self.model_fine(curr_x, t_tensor, prompt_emb)
                
                # Scheduler Step (eta > 0 allowed here for stochastic sampling)
                prev_x, _, _ = self.scheduler.step(
                    noise_pred, t.item(), curr_x, 
                    eta=self.config.eta, # Usually 1.0 for training exploration
                    num_inference_steps=self.config.num_inference_steps
                )
                
                traj.append(prev_x)
                curr_x = prev_x
                
        # traj[0] is Noise, traj[-1] is Data (X_final)
        
        # --- 2. Adjoint Initialization ---
        # "Noiseless update" trick from Paper Appendix H.1 to reduce variance.
        # Use the state X_{K-1} (second to last) to predict clean X_0/Step.
        
        x_prev = traj[-2] 
        t_last = timesteps[-1].item() # t=0
        
        with torch.enable_grad():
            x_prev = x_prev.detach().requires_grad_(True)
            # Use model_fine for this final clean prediction
            t_tensor = torch.tensor([t_last], device=device).repeat(batch_size)
            noise_pred = self.model_fine(x_prev, t_tensor, prompt_emb)
            
            # Deterministic step (eta=0)
            x_final_clean, _, _ = self.scheduler.step(
                noise_pred, t_last, x_prev, eta=0.0,
                num_inference_steps=self.config.num_inference_steps
            )
            
            # Compute Reward Gradient
            # Note: reward_multiplier is usually positive in config, 
            # but am_trainer.py line 125 negates it: a = - multiplier * grad.
            # In S-MEME: Reward = Entropy. Gradient = -Score.
            # So if reward_fn returns entropy variation, we want gradient ascent.
            # Adjoint usually points in direction of improvement.
            
            r = reward_fn(x_final_clean).sum()
            r.backward()
            reward_grad = x_final_clean.grad.detach()
            
        # Initialize Adjoint State (Line 125 am_trainer.py)
        # "a = -self.config.reward_multiplier * reward_grads"
        adjoint_state = -self.config.reward_multiplier * reward_grad
        
        # Store adjoints for loss computation
        # Mapping: timestep index -> adjoint vector
        adjoint_storage = {} 
        
        # --- 3. Backward Pass (Lean Adjoint ODE) ---
        # Propagate adjoint from Data -> Noise
        # timesteps are [975, 950, ..., 0]
        # We iterate backwards through the TRAJECTORY array.
        # traj index k corresponds to timesteps[k]
        
        # Loop K-1 down to 0
        for k in range(self.config.num_inference_steps - 1, -1, -1):
            t_val = timesteps[k].item()
            x_k = traj[k] # State at time t
            
            # Calculate VJP (Eq 220 / am_trainer line 129)
            # grad_inner_product uses model_pre (base dynamics)
            vjp = self._grad_inner_product(x_k, t_val, adjoint_state, prompt_emb)
            
            # Euler update: a_{t-1} = a_t + grad
            # (Note: directions might be swapped depending on t definition, 
            # but code adds: a += grad_inner_prod)
            adjoint_state = adjoint_state + vjp
            
            # Store
            adjoint_storage[k] = adjoint_state
            
        # --- 4. Loss Computation ---
        # Sample subset of indices using Stratified Sampling (Repo Logic)
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
            
            # Re-run models to get gradients
            t_tensor = torch.tensor([t_val], device=device).repeat(batch_size)
            eps_fine = self.model_fine(x_k, t_tensor, prompt_emb)
            with torch.no_grad():
                eps_pre = self.model_pre(x_k, t_tensor, prompt_emb)
                
            # Compute Coefficients for Loss Scaling (Eq 222)
            # We need standard deviation sigma_t for the scaling
            # The code in compute_loss uses:
            # loss = (control_times_sqrt_dt + adjoint * std_dev)^2
            # effectively: loss = ( (eps_fine - eps_pre) + adjoint * sigma_t )^2
            
            # Get sigma_t from scheduler logic (re-using step logic or formula)
            # Based on am_trainer logic: "std_dev_t" comes from evaluate_controls
            # which gets it from the scheduler output.
            
            # Let's calculate std_dev_t for this step
            # Note: The scheduler step gives sigma for the NEXT step. 
            # We need sigma for current step k.
            _, _, std_dev_t = self.scheduler.step(eps_fine, t_val, x_k, eta=self.config.eta, 
                                                  num_inference_steps=self.config.num_inference_steps)
            
            # Scaling: The paper Eq 222 is complex, but the code simplifies it:
            # It seems to treat the control difference directly in noise space
            # and scales the adjoint into noise space using sigma_t.
            
            # Code: loss = ( (eps_fine - eps_pre) + target_adjoint * std_dev_t )^2
            # Note: std_dev_t handles the "dt" scaling implicitly.
            
            diff = (eps_fine - eps_pre)
            target = target_adjoint * std_dev_t.view(-1, 1) # broadcasting
            
            # Squared error per sample (sum over dimensions)
            loss_per_sample = torch.sum((diff + target) ** 2, dim=1)
            
            # --- 5. EMA Quantile Clipping (The SOTA Trick) ---
            # am_trainer.py lines 272-288
            
            # 1. Gather all loss values (if distributed, skipping here)
            loss_root = torch.sqrt(loss_per_sample.detach())
            
            # 2. Compute quantile (e.g. 0.9)
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            
            # 3. Update EMA
            ema_threshold = self._ema_update(current_quantile)
            
            # 4. Mask
            mask = (loss_root < ema_threshold).float()
            
            # 5. Apply
            loss_final = (loss_per_sample * mask).mean()
            
            total_loss += loss_final
            
        return total_loss
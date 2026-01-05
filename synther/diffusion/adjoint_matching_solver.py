import torch
import torch.nn as nn
import numpy as np

class FlowMatchingSolver:
    """
    Implements a simple Euler ODE solver for Flow Matching physics.
    """
    def __init__(self, num_inference_steps=20):
        self.num_inference_steps = num_inference_steps
        # Avoid t=1.0 singularity by stopping at 0.99
        self.timesteps = torch.linspace(0, 0.99, num_inference_steps + 1)

    def get_velocity(self, model_wrapper, x, t):
        return model_wrapper(x, t)

    def step_euler(self, x, v, dt):
        return x + v * dt

class AdjointMatchingSolver:
    def __init__(self, model_pre, model_fine, config):
        self.model_pre = model_pre
        self.model_fine = model_fine
        self.config = config
        self.solver = FlowMatchingSolver(num_inference_steps=config.num_inference_steps)
        
        # EMA for Quantile Clipping
        self.ema_quantile = None
        self.ema_decay = 0.99

    def _ema_update(self, new_val):
        if self.ema_quantile is None:
            self.ema_quantile = new_val
        else:
            self.ema_quantile = self.ema_decay * self.ema_quantile + (1 - self.ema_decay) * new_val
        return self.ema_quantile

    def solve_and_compute_grad(self, x_start, prompt_emb, target_grad_fn, active_indices):
        device = x_start.device
        batch_size = x_start.shape[0]
        dt = 1.0 / self.config.num_inference_steps
        
        # 1. Forward Pass
        traj = [x_start.detach()]
        x_t = x_start.detach()
        times = self.solver.timesteps.to(device)
        
        for i in range(self.config.num_inference_steps):
            t_now = times[i]
            with torch.no_grad():
                v_fine = self.solver.get_velocity(self.model_fine, x_t, t_now)
            
            # Sanity Check
            if torch.isnan(v_fine).any() or v_fine.abs().max() > 1e4:
                return torch.tensor(0.0, device=device, requires_grad=True)

            x_next = self.solver.step_euler(x_t, v_fine, dt)
            traj.append(x_next.detach())
            x_t = x_next
            
        x_final = traj[-1]
        
        # 2. Compute Reward Gradient (Normalized)
        reward_grad = target_grad_fn(x_final)
        grad_norm = torch.norm(reward_grad, dim=1, keepdim=True) + 1e-8
        reward_grad = reward_grad / grad_norm 
        
        lambda_t = -1.0 * reward_grad * self.config.reward_multiplier
        
        # 3. Backward Pass (Solve Adjoint ODE)
        adjoint_storage = {} 
        
        for i in reversed(range(self.config.num_inference_steps)):
            t_now = times[i]
            x_curr = traj[i]
            adjoint_storage[i] = lambda_t.detach()
            
            with torch.enable_grad():
                x_curr.requires_grad_(True)
                v_curr = self.solver.get_velocity(self.model_fine, x_curr, t_now)
                # VJP Calculation
                vjp = torch.autograd.grad(v_curr, x_curr, grad_outputs=lambda_t, retain_graph=False)[0]
                
            vjp = torch.clamp(vjp, min=-5.0, max=5.0)
            lambda_t = lambda_t + vjp * dt
            lambda_t = torch.clamp(lambda_t, min=-5.0, max=5.0)

        # 4. Compute Loss
        total_loss = 0.0
        valid_samples = 0
        
        for k_idx in active_indices:
            k = k_idx.item()
            t_val = times[k]
            x_k = traj[k].detach()
            target_adjoint = adjoint_storage[k].detach()
            
            v_fine = self.solver.get_velocity(self.model_fine, x_k, t_val)
            with torch.no_grad():
                v_pre = self.solver.get_velocity(self.model_pre, x_k, t_val)
                
            diff = (v_fine - v_pre)
            loss_per_sample = torch.sum((diff + target_adjoint) ** 2, dim=1)
            
            loss_root = torch.sqrt(loss_per_sample) # Keep grad here? No, loss_per_sample has grad
            
            if torch.isnan(loss_root).any():
                continue
            
            # Detach for quantile calculation
            current_quantile = torch.quantile(loss_root.detach(), self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            
            # [FIXED] Use <= and add epsilon to ensure we don't drop identical samples
            # This prevents the "All Zero" mask when loss is constant
            mask = (loss_root <= (ema_threshold + 1e-6)).float()
            
            total_loss += torch.mean(loss_per_sample * mask)
            valid_samples += 1
            
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return total_loss / valid_samples
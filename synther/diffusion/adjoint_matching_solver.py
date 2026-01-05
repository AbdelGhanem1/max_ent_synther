import torch
import torch.nn as nn
import numpy as np

class FlowMatchingSolver:
    """
    Implements a simple Euler ODE solver for Flow Matching physics (t=0 -> t=1).
    Replaces DDIMScheduler to ensure compatibility with S-MEME math.
    """
    def __init__(self, num_inference_steps=20):
        self.num_inference_steps = num_inference_steps
        # Simple linear time steps [0, ..., 1]
        self.timesteps = torch.linspace(0, 1, num_inference_steps + 1)

    def get_velocity(self, model_wrapper, x, t):
        """Queries the model to get the Vector Field v(x, t)."""
        return model_wrapper(x, t)

    def step_euler(self, x, v, dt):
        """Simple Euler integration step."""
        return x + v * dt

class AdjointMatchingSolver:
    def __init__(self, model_pre, model_fine, config):
        self.model_pre = model_pre
        self.model_fine = model_fine
        self.config = config
        self.solver = FlowMatchingSolver(num_inference_steps=config.num_inference_steps)
        
        # EMA for Quantile Clipping (stabilizes training)
        self.ema_quantile = 0.0
        self.ema_decay = 0.99

    def _ema_update(self, new_val):
        self.ema_quantile = self.ema_decay * self.ema_quantile + (1 - self.ema_decay) * new_val
        return self.ema_quantile

    def solve_and_compute_grad(self, x_start, prompt_emb, target_grad_fn, active_indices):
        """
        Performs the Forward Pass (Generation) and Backward Pass (Adjoint Calculation).
        
        Args:
            target_grad_fn: A function that returns the Gradient of the Reward at x_final.
                            For S-MEME, this returns -Score(x).
        """
        device = x_start.device
        batch_size = x_start.shape[0]
        dt = 1.0 / self.config.num_inference_steps
        
        # 1. Forward Pass (Generate Trajectory)
        traj = [x_start.detach()]
        x_t = x_start.detach()
        times = self.solver.timesteps.to(device)
        
        for i in range(self.config.num_inference_steps):
            t_now = times[i]
            with torch.no_grad():
                v_fine = self.solver.get_velocity(self.model_fine, x_t, t_now)
            x_next = self.solver.step_euler(x_t, v_fine, dt)
            traj.append(x_next.detach())
            x_t = x_next
            
        x_final = traj[-1]
        
        # 2. Compute Adjoint Terminal Condition
        # We need the gradient of the Reward. 
        # For S-MEME, we want to MAXIMIZE Entropy => Move against Density.
        # Target Gradient = -Score(x).
        
        # Calculate the gradient at x_final
        reward_grad = target_grad_fn(x_final)
        
        # Adjoint State lambda = - Gradient * Multiplier
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
                # Vector-Jacobian Product: lambda^T * dv/dx
                vjp = torch.autograd.grad(v_curr, x_curr, grad_outputs=lambda_t, retain_graph=False)[0]
                
            # Update lambda (Euler step backwards)
            lambda_t = lambda_t + vjp * dt

        # 4. Compute Matching Loss
        total_loss = 0.0
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
            
            # Robust Clipping
            loss_root = torch.sqrt(loss_per_sample.detach())
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            mask = (loss_root < ema_threshold).float()
            
            total_loss += torch.mean(loss_per_sample * mask)
            
        return total_loss / len(active_indices)
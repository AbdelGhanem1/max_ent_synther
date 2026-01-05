import torch
import torch.nn as nn
import numpy as np

class FlowMatchingSolver:
    def __init__(self, num_inference_steps=20):
        self.num_inference_steps = num_inference_steps
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
            # [Check] Ensure we are tracking gradients if model_fine is involved?
            # Actually, forward pass is no_grad for trajectory, 
            # but we need graph for backward pass later.
            with torch.no_grad():
                v_fine = self.solver.get_velocity(self.model_fine, x_t, t_now)
            
            # Sanity Check
            if torch.isnan(v_fine).any() or v_fine.abs().max() > 1e4:
                return torch.tensor(0.0, device=device, requires_grad=True)

            x_next = self.solver.step_euler(x_t, v_fine, dt)
            traj.append(x_next.detach())
            x_t = x_next
            
        x_final = traj[-1]
        
        # 2. Compute Reward Gradient
        reward_grad = target_grad_fn(x_final)
        
        # Normalize to stabilize
        grad_norm = torch.norm(reward_grad, dim=1, keepdim=True) + 1e-8
        reward_grad = reward_grad / grad_norm 
        
        lambda_t = -1.0 * reward_grad * self.config.reward_multiplier
        
        # 3. Backward Pass
        adjoint_storage = {} 
        
        for i in reversed(range(self.config.num_inference_steps)):
            t_now = times[i]
            x_curr = traj[i]
            
            # Store lambda for loss
            adjoint_storage[i] = lambda_t.detach()
            
            with torch.enable_grad():
                x_curr.requires_grad_(True)
                v_curr = self.solver.get_velocity(self.model_fine, x_curr, t_now)
                
                # VJP: lambda^T * dv/dx
                vjp = torch.autograd.grad(v_curr, x_curr, grad_outputs=lambda_t, retain_graph=False)[0]
                
            vjp = torch.clamp(vjp, min=-5.0, max=5.0)
            lambda_t = lambda_t + vjp * dt
            lambda_t = torch.clamp(lambda_t, min=-5.0, max=5.0)

        # 4. Compute Loss
        total_loss = 0.0
        
        for k_idx in active_indices:
            k = k_idx.item()
            t_val = times[k]
            x_k = traj[k].detach()
            target_adjoint = adjoint_storage[k].detach()
            
            # Enable Gradients here for the parameter update
            v_fine = self.solver.get_velocity(self.model_fine, x_k, t_val)
            
            with torch.no_grad():
                v_pre = self.solver.get_velocity(self.model_pre, x_k, t_val)
                
            diff = (v_fine - v_pre)
            
            # Standard Adjoint Matching Loss
            # We remove the sophisticated masking to ensure signal flows
            loss_per_sample = torch.sum((diff + target_adjoint) ** 2, dim=1)
            
            total_loss += torch.mean(loss_per_sample)
            
        return total_loss / len(active_indices)
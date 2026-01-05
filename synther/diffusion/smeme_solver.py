import torch
import copy
import numpy as np
from tqdm import tqdm
import wandb
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from torch.amp import autocast

# --- 1. The Corrected Wrapper (Manual Karras Scaling) ---
class EDMToFlowWrapper(torch.nn.Module):
    """
    Wraps an EDM model to behave like a Flow Matching model.
    CRITICAL FIX: Implements Karras Preconditioning manually.
    We cannot call model(x) because that returns loss. 
    We must call model.net(c_in * x) and scale the output.
    """
    def __init__(self, edm_model, sigma_min=0.002, sigma_max=80.0, sigma_data=1.0):
        super().__init__()
        self.edm_model = edm_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
    def forward(self, x, t):
        # 1. Map t (0=Noise, 1=Data) to Sigma (High -> Low)
        if isinstance(t, float):
            t = torch.tensor(t, device=x.device)
        
        # Clamp t to avoid div by zero at t=1
        t_safe = torch.clamp(t, 0.0, 0.999)
        
        # Linear schedule: t=0 -> sigma_max, t=1 -> sigma_min
        # (This aligns Flow Matching time with EDM noise levels)
        sigma = self.sigma_max * (1.0 - t_safe)
        
        # Broadcasting
        if sigma.ndim == 0:
            sigma = sigma.view(1).repeat(x.shape[0])
            
        # 2. Karras Preconditioning Math (Eq. 7 in Karras 2022)
        # We need this to turn the raw neural net output into D(x)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = 0.25 * sigma.log()
        
        # 3. Call the Inner Network
        # We assume self.edm_model.net is the actual MLP/UNet
        # Ensure inputs are float32 for stability
        x_in = (c_in.view(-1, 1) * x).float()
        sigma_in = c_noise.view(-1).float()
        
        F_x = self.edm_model.net(x_in, sigma_in).float()
        
        # 4. Reconstruct Denoised Image D(x)
        D_x = c_skip.view(-1, 1) * x + c_out.view(-1, 1) * F_x
        
        # 5. Convert to Velocity v = (D_x - x) / (1-t)
        # Flow Matching ODE: dx/dt = v
        # x_t = (1-t)x_noise + t*x_data
        # v = x_data - x_noise = (D_x - x) / (1 - t)
        v = (D_x - x) / (1.0 - t_safe)
        
        return v

class SMEMESolver:
    def __init__(self, base_model, config):
        self.current_model = base_model
        self.config = config
        
        self.previous_model = copy.deepcopy(base_model)
        self.previous_model.eval()
        self.previous_model.requires_grad_(False)
        
        self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=1e-5, weight_decay=1e-2)

    def _get_score_at_data(self, model, x_data):
        """
        Computes Score using the wrapped model to ensure consistency.
        Score approx (D(x) - x) / sigma^2
        """
        # Small sigma for score approximation
        sigma_val = 0.05
        sigma = torch.tensor(sigma_val, device=x_data.device).repeat(x_data.shape[0])
        
        # Use our wrapper logic manually for the score
        # (Or just instantiate a temporary wrapper)
        wrapper = EDMToFlowWrapper(model)
        
        # To get D(x) from the wrapper at this sigma, we need the t that corresponds to it
        # sigma = sigma_max * (1-t) => t = 1 - sigma/sigma_max
        t_equiv = 1.0 - (sigma_val / wrapper.sigma_max)
        
        # Get Velocity from wrapper
        v = wrapper(x_data, t_equiv)
        
        # Unwrap Velocity to get D(x): D_x = v * (1-t) + x
        # approx D_x - x = v * (1-t)
        diff = v * (1.0 - t_equiv)
        
        # Score = (D_x - x) / sigma^2
        score = diff / (sigma_val ** 2)
        score = torch.clamp(score, min=-100.0, max=100.0)
                
        return score

    def train(self, data_loader):
        device = next(self.current_model.parameters()).device
        
        for iteration in range(self.config.num_smeme_iterations):
            print(f"=== S-MEME Iteration {iteration + 1} ===")
            
            # Wrap models
            model_pre_wrapped = EDMToFlowWrapper(self.previous_model)
            model_fine_wrapped = EDMToFlowWrapper(self.current_model)
            
            am_solver = AdjointMatchingSolver(
                model_pre=model_pre_wrapped,
                model_fine=model_fine_wrapped,
                config=self.config.am_config
            )

            # Define Reward Gradient (Maximize Surprise -> Move against Score)
            def smeme_reward_grad_fn(x):
                score = self._get_score_at_data(self.previous_model, x)
                return -1.0 * score 

            pbar = tqdm(range(self.config.steps_per_iter)) 
            
            for _ in pbar:
                try:
                    x_start = next(data_loader).to(device)
                    self.optimizer.zero_grad()
                    
                    loss = am_solver.solve_and_compute_grad(
                        x_start=x_start,
                        prompt_emb=None,
                        target_grad_fn=smeme_reward_grad_fn, 
                        active_indices=torch.tensor(range(self.config.am_config.num_inference_steps))
                    )
                    
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                    if wandb.run:
                        wandb.log({"am_loss": loss.item()})
                    
                except StopIteration:
                    break

            self.previous_model.load_state_dict(self.current_model.state_dict())
            
        return self.current_model
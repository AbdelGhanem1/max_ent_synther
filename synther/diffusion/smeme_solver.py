import torch
import copy
import numpy as np
from tqdm import tqdm
import wandb
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from torch.amp import autocast

# --- Helper: EDM Physics Wrapper ---
class EDMToFlowWrapper(torch.nn.Module):
    """
    Wraps an EDM (Karras) model to behave like a Flow Matching model.
    Maps t=[0,1] (Flow) -> sigma=[max, min] (EDM).
    """
    def __init__(self, edm_model, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.edm_model = edm_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def forward(self, x, t):
        # Map t (0=Noise, 1=Data) to Sigma (High -> Low)
        if isinstance(t, float):
            t = torch.tensor(t, device=x.device)
        
        t_safe = torch.clamp(t, 0.0, 0.999) 
        sigma = self.sigma_max * (1.0 - t_safe)
        
        if sigma.ndim == 0:
            sigma = sigma.view(1).repeat(x.shape[0])
            
        # Get EDM Output (Denoised Estimate x_0)
        D_x = self.edm_model(x, sigma)
        
        # Convert to Velocity v = (D_x - x) / (1-t)
        v = (D_x - x) / (1.0 - t_safe)
        return v

class SMEMESolver:
    def __init__(self, base_model, config):
        self.current_model = base_model
        self.config = config
        
        self.previous_model = copy.deepcopy(base_model)
        self.previous_model.eval()
        self.previous_model.requires_grad_(False)
        
        # Lower LR is crucial for fine-tuning
        self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=1e-5, weight_decay=1e-2)

    def _get_score_at_data(self, model, x_data):
        """
        Computes the Score (Gradient of Log Likelihood) using EDM Physics.
        Score = (D(x) - x) / sigma^2 approx -epsilon / sigma
        For clean data (t=1, sigma approx 0), we approximate using a small sigma.
        """
        # We use a small sigma to approximate the score at the manifold
        sigma_val = 0.05 
        sigma = torch.tensor(sigma_val, device=x_data.device).repeat(x_data.shape[0])
        
        with torch.no_grad():
             # Run model in whatever precision it wants
            with autocast(device_type='cuda', enabled=False):
                # Force inputs to float32 for precision
                x_in = x_data.float()
                sigma_in = sigma.float().view(-1)
                
                D_x = model(x_in, sigma_in).float()
                
                # EDM Score Formula: (D(x) - x) / sigma^2
                score = (D_x - x_in) / (sigma_val ** 2)
                
                # Clip to prevent explosions
                score = torch.clamp(score, min=-100.0, max=100.0)
                
        return score

    def train(self, data_loader):
        device = next(self.current_model.parameters()).device
        
        for iteration in range(self.config.num_smeme_iterations):
            print(f"=== S-MEME Iteration {iteration + 1} ===")
            
            # 1. Update wrappers with current models
            model_pre_wrapped = EDMToFlowWrapper(self.previous_model)
            model_fine_wrapped = EDMToFlowWrapper(self.current_model)
            
            # 2. Init Solver
            am_solver = AdjointMatchingSolver(
                model_pre=model_pre_wrapped,
                model_fine=model_fine_wrapped,
                config=self.config.am_config
            )

            # 3. Define the S-MEME Reward Gradient Function
            # We want to MAXIMIZE Surprise (Entropy).
            # Surprise = - log p(x). Gradient = - Score(x).
            # This function returns the direction we want to push the particles.
            def smeme_reward_grad_fn(x):
                score = self._get_score_at_data(self.previous_model, x)
                # We want to move AWAY from high probability.
                # So we follow -Score.
                return -1.0 * score 

            pbar = tqdm(range(self.config.steps_per_iter)) 
            
            for _ in pbar:
                try:
                    # Sample Pure Noise (t=0)
                    x_start = next(data_loader).to(device)
                    self.optimizer.zero_grad()
                    
                    # 4. Run Adjoint Matching
                    loss = am_solver.solve_and_compute_grad(
                        x_start=x_start,
                        prompt_emb=None,
                        target_grad_fn=smeme_reward_grad_fn, 
                        active_indices=torch.tensor(range(self.config.am_config.num_inference_steps))
                    )
                    
                    # 5. Update
                    # Gradients are already computed by solve_and_compute_grad via autograd
                    # We just need to step the optimizer for self.current_model
                    
                    # Note: solve_and_compute_grad doesn't call backward() on the loss itself 
                    # if it returns a detached float.
                    # Wait, my AdjointSolver implementation returns a differentiable loss.
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    pbar.set_description(f"Loss: {loss.item():.4f}")
                    if wandb.run:
                        wandb.log({"am_loss": loss.item()})
                    
                except StopIteration:
                    break

            # End of Iteration: Update Reference Model
            self.previous_model.load_state_dict(self.current_model.state_dict())
            
        return self.current_model
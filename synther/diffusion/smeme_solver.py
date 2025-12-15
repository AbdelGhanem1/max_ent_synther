import torch
import copy
import numpy as np
from tqdm import tqdm
import wandb
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from torch.cuda.amp import autocast, GradScaler # <--- NEW: For Mixed Precision

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
        
        # --- OPTIMIZATION 1: PyTorch 2.0 Compilation ---
        # If available, we compile the underlying backbone model.
        # This fuses kernels and speeds up the tight ODE loop significantly.
        if hasattr(torch, "compile"):
            print("⚡ PyTorch 2.0 detected. Compiling models for speed...")
            try:
                # We compile the .model attribute (the actual ResMLP/UNet)
                # mode='reduce-overhead' is best for small batches/loops
                self.current_model.model = torch.compile(self.current_model.model, mode='reduce-overhead')
                self.previous_model.model = torch.compile(self.previous_model.model, mode='reduce-overhead')
            except Exception as e:
                print(f"⚠️ Compilation failed (continuing without): {e}")

    def _get_score_at_data(self, model, x_data, t_idx):
        """
        Computes a STABLE Score Proxy s(x, t).
        Returns -epsilon (Stable Direction).
        """
        t_tensor = torch.tensor([t_idx], device=x_data.device).repeat(x_data.shape[0])
        
        with torch.no_grad():
            with autocast(): # <--- NEW: Run inference in FP16
                eps = model(x_data, t_tensor)
                score = -eps 
            
        return score

    def train(self, train_loader):
        print(f"Starting S-MEME with K={self.config.num_smeme_iterations} iterations.")
        
        # --- OPTIMIZATION 2: GradScaler for AMP ---
        scaler = GradScaler()
        
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
                # Returns Positive Score (because Solver applies negative sign)
                score = self._get_score_at_data(self.previous_model, x, t_idx)
                return score 
            
            pbar = tqdm(train_loader, desc=f"Iter {k+1}", dynamic_ncols=True)
            running_loss = 0.0
            
            for step, batch in enumerate(pbar):
                self.current_model.zero_grad()
                
                # --- OPTIMIZATION 3: Mixed Precision Context ---
                # Runs the entire ODE solve (Forward + Adjoint) in FP16/BF16
                with autocast():
                    loss = solver.solve_vector_field(batch, entropy_gradient_fn)
                
                # --- OPTIMIZATION 4: Scaled Backward Pass ---
                # Prevents underflow of small gradients in FP16
                scaler.scale(loss).backward()
                
                # Unscale before clipping
                if not hasattr(self, 'optimizer'):
                    self.optimizer = torch.optim.AdamW(self.current_model.parameters(), lr=1e-4, weight_decay=1e-2)
                
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                
                # Step with scaler
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
            
            # The vector field calc might need autocast context too, but usually inherits
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
            target = target_adjoint * std_dev_t.view(-1, 1)
            loss_per_sample = torch.sum((diff + target) ** 2, dim=1)
            
            # EMA Clipping
            loss_root = torch.sqrt(loss_per_sample.detach())
            current_quantile = torch.quantile(loss_root, self.config.per_sample_threshold_quantile)
            ema_threshold = self._ema_update(current_quantile)
            mask = (loss_root < ema_threshold).float()
            
            total_loss += (loss_per_sample * mask).mean()
            
        return total_loss
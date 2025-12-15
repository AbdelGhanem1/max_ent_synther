import torch
import copy
from adjoint_matching_solver import AdjointMatchingSolver

class SMEMESolver:
    def __init__(self, base_model, config):
        """
        Implements S-MEME (Algorithm 1) using Adjoint Matching as the linear solver.
        
        Args:
            base_model: The pre-trained diffusion model (noise predictor).
            config: Configuration object containing:
                - num_smeme_iterations (K): e.g., 3 or 4
                - alpha_schedule: list of regularization strengths [alpha_1, ..., alpha_K]
                - am_config: Configuration for the inner AdjointMatchingSolver
        """
        self.current_model = base_model
        self.config = config
        
        # We need a deep copy of the starting point to serve as the "previous" model
        # in the first iteration.
        self.previous_model = copy.deepcopy(base_model)
        self.previous_model.eval()
        self.previous_model.requires_grad_(False)

    def _get_score_at_data(self, model, x_data, t_idx):
        """
        Computes the Score Function s(x, t) at the data level.
        
        Equation: s(x, t) = -epsilon(x, t) / sqrt(1 - alpha_bar_t)
        
        Crucial Engineering Detail:
        Directly evaluating score at t=0 (data) is numerically unstable because
        sqrt(1 - alpha_bar_0) is 0 or very small.
        
        We rely on the "Noiseless Trick" context from the solver:
        The solver passes us a 'clean' x computed from the penultimate step.
        We evaluate the score using the noise prediction parameters for that step.
        """
        # We need the scheduler to get alpha_bar
        # We can borrow the scheduler from the solver instance later, 
        # or create a temporary one.
        # Assuming the model wrapper handles t_idx correctly.
        
        # In S-MEME, we use the score of the PREVIOUS model as the gradient.
        # The solver provides x_final_clean.
        
        t_tensor = torch.tensor([t_idx], device=x_data.device).repeat(x_data.shape[0])
        
        with torch.no_grad():
            # 1. Predict Noise
            eps = model(x_data, t_tensor)
            
            # 2. Get Scaling Factor (1 / sqrt(1 - alpha_bar))
            # We need the alpha_bar for this specific timestep.
            # Ideally, access the scheduler from the solver.
            # For now, we compute it assuming the standard linear schedule provided in config.
            # (In a real codebase, pass the scheduler instance).
            
            # Replicating scheduler logic for alpha_bar
            num_train_timesteps = self.config.am_config.num_train_timesteps
            betas = torch.linspace(0.0001, 0.02, num_train_timesteps, device=x_data.device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            alpha_bar = alphas_cumprod[t_idx]
            
            # Score = -eps / sqrt(1 - alpha_bar)
            # Add epsilon to denominator for stability
            score = -eps / torch.sqrt(1 - alpha_bar + 1e-6)
            
        return score

    def train(self, train_loader):
        """
        Main S-MEME Loop (Algorithm 1)
        """
        print(f"Starting S-MEME with K={self.config.num_smeme_iterations} iterations.")
        
        for k in range(self.config.num_smeme_iterations):
            print(f"=== S-MEME Iteration {k+1}/{self.config.num_smeme_iterations} ===")
            
            # 1. Setup the Linear Solver for this iteration
            # We use a subclass that accepts a vector field instead of a scalar reward
            
            # Get alpha for this step (Regularization coefficient)
            # Reward Multiplier in Adjoint Matching is 1/alpha
            alpha = self.config.alpha_schedule[k]
            self.config.am_config.reward_multiplier = 1.0 / alpha
            
            solver = VectorFieldAdjointSolver(
                model_pre=self.previous_model, # Regularization target (Previous model)
                model_fine=self.current_model, # Model being updated
                config=self.config.am_config
            )
            
            # 2. Define the "Reward Gradient" function
            # This corresponds to line 3 in Algorithm 1: "Set grad f_k = -s^{k-1}"
            # The solver expects a function that takes (x, t_idx) and returns a vector.
            def entropy_gradient_fn(x, t_idx):
                # Gradient of Entropy = -Score
                # But we want Gradient Ascent on Reward.
                # Reward = Entropy. 
                # Gradient of Reward = Gradient of Entropy = -Score.
                # So we return -Score.
                
                score = self._get_score_at_data(self.previous_model, x, t_idx)
                return -score # This is the gradient vector
            
            # 3. Inner Loop: Solve the Linear Fine-tuning Problem
            # Iterate over data batches (or just noise batches)
            # S-MEME generates its own data starting from noise.
            
            for step, batch in enumerate(train_loader):
                # x_0 is pure noise here
                noise = torch.randn_like(batch) 
                
                # Zero grad
                self.current_model.zero_grad()
                
                # Execute Solver
                loss = solver.solve_vector_field(noise, entropy_gradient_fn)
                
                # Update Weights
                loss.backward()
                # Gradient clipping is standard in diffusion training
                torch.nn.utils.clip_grad_norm_(self.current_model.parameters(), 1.0)
                # optimizer.step() (Assuming optimizer is managed externally or passed in)
                
                # Log progress...
                
            # 4. Update "Previous Model"
            # Algorithm 1 Line 5: pi_{k-1} <- pi_k
            # We freeze the current state of the model to serve as the reference for the next step.
            self.previous_model.load_state_dict(self.current_model.state_dict())
            self.previous_model.eval()
            self.previous_model.requires_grad_(False)
            
        return self.current_model


class VectorFieldAdjointSolver(AdjointMatchingSolver):
    """
    Subclass of AdjointMatchingSolver that overrides the initialization step.
    Instead of calculating grad(scalar_reward), it accepts a direct vector field.
    """
    
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        """
        Modified solve method for S-MEME.
        Args:
            vector_field_fn: Function(x, t_idx) -> Tensor (Gradient Vector)
        """
        # ... [Reuse initialization logic from parent] ...
        device = x_0.device
        batch_size = x_0.shape[0]
        timesteps = torch.linspace(
            self.config.num_train_timesteps - 1, 0, 
            self.config.num_inference_steps, 
            device=device, dtype=torch.long
        )
        
        # 1. Forward Pass (Same as parent)
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

        # 2. Adjoint Initialization (OVERRIDDEN)
        x_prev = traj[-2]
        t_last = timesteps[-1].item()
        
        # Noiseless update to get clean x
        # Note: We don't need gradients on x_prev here because we aren't backpropagating 
        # through a scalar reward function. We just need the value x_final_clean 
        # to query the vector field.
        with torch.no_grad():
            t_tensor = torch.tensor([t_last], device=device).repeat(batch_size)
            noise_pred = self.model_fine(x_prev, t_tensor, prompt_emb)
            x_final_clean, _, _ = self.scheduler.step(
                noise_pred, t_last, x_prev, eta=0.0,
                num_inference_steps=self.config.num_inference_steps
            )
            
            # --- DIFFERENCE IS HERE ---
            # Instead of: r = reward(x); r.backward()
            # We call the vector field function directly.
            
            # Get the gradient vector (Negative Score) from the previous model
            reward_grad = vector_field_fn(x_final_clean, t_last)
            
        # Continue with standard logic
        # Initialize Adjoint State
        adjoint_state = -self.config.reward_multiplier * reward_grad
        
        # 3. Backward Pass (Same as parent)
        adjoint_storage = {}
        for k in range(self.config.num_inference_steps - 1, -1, -1):
            t_val = timesteps[k].item()
            x_k = traj[k]
            vjp = self._grad_inner_product(x_k, t_val, adjoint_state, prompt_emb)
            adjoint_state = adjoint_state + vjp
            adjoint_storage[k] = adjoint_state
            
        # 4. Loss Computation (Same as parent)
        # ... [Call parent loss logic or copy-paste if reuse is hard] ...
        # For conciseness, assuming we can reuse or copy the loss block here.
        
        # (Copying the loss block from previous response to ensure self-contained correctness)
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
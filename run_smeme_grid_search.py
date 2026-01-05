import torch
import numpy as np
import copy
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=5000):
    # Same 70/30 setup
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_a, 2))
    data_b = np.random.normal(loc=[2.0, 2.0], scale=1.0, size=(n_samples - n_a, 2))
    return np.vstack([data_a, data_b]).astype(np.float32)

def check_health(model, step_name):
    """Returns True if healthy, False if NaN/Inf detected"""
    # 1. Check Parameters (Weights)
    max_w = 0.0
    has_nan = False
    for name, p in model.named_parameters():
        if p.requires_grad:
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"!!! EXPLOSION DETECTED in Weights ({name}) at {step_name} !!!")
                has_nan = True
            max_w = max(max_w, p.abs().max().item())
            
    # 2. Check Gradients
    max_g = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"!!! EXPLOSION DETECTED in Gradients ({name}) at {step_name} !!!")
                has_nan = True
            max_g = max(max_g, p.grad.abs().max().item())

    return has_nan, max_w, max_g

def run_debug():
    device = get_device()
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    print(">>> 1. Training Base Model (Fast)...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    for _ in range(1000): # Short pre-train just to get started
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()

    print("\n>>> 2. Running S-MEME Stress Test (Alpha=1.0)...")
    # We use the settings that caused the 'disappearance'
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=1.0) 
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    # UN-PATCHED SOLVER (We want to see the error)
    # If you were using the monkey-patch, remove it or ensure this uses the raw logic.
    # Assuming standard behavior here to catch the bug.
    
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    print(f"{'Step':<10} | {'Max Weight':<15} | {'Max Grad':<15} | {'Status'}")
    print("-" * 50)
    
    for step in range(2001):
        # Forward & Backward
        noise = torch.randn(256, 2).to(device)
        opt_s.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        
        # Check Health BEFORE Step
        is_nan, max_w, max_g = check_health(current_model, f"Step {step}")
        
        if step % 100 == 0 or is_nan:
            status = "CRITICAL FAIL" if is_nan else "OK"
            print(f"{step:<10} | {max_w:<15.4f} | {max_g:<15.4f} | {status}")
            
        if is_nan:
            print("\n>>> DIAGNOSIS: The model weights or gradients became NaN.")
            print(">>> CAUSE: This confirms the disappearance is due to numerical explosion.")
            break
            
        opt_s.step()

if __name__ == "__main__":
    run_debug()
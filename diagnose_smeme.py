import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

# Imports from your current setup
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.smeme_solver import EDMToFlowWrapper, SMEMESolver
from synther.diffusion.adjoint_matching_solver import FlowMatchingSolver, AdjointMatchingSolver

def diagnose_smeme_failure():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Running S-MEME Diagnostics on {device}")

    # 1. Create Synthetic Data (90% A, 10% B)
    # ---------------------------------------------------------
    print("\n[1] Creating Dummy Data...")
    n = 1000
    # Cluster A (Common)
    data_a = np.random.normal(loc=[-0.5, -0.5], scale=0.1, size=(int(0.9*n), 2))
    # Cluster B (Rare)
    data_b = np.random.normal(loc=[0.5, 0.5], scale=0.1, size=(int(0.1*n), 2))
    dataset = torch.from_numpy(np.vstack([data_a, data_b]).astype(np.float32)).to(device)

    # 2. Train Base Model
    # ---------------------------------------------------------
    print("[2] Training Base Model (Fast)...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train briefly just to establish the modes
    for _ in range(500):
        loss = base_model(dataset)
        opt.zero_grad(); loss.backward(); opt.step()

    # 3. Diagnose: The Wrapper & Velocity
    # ---------------------------------------------------------
    print("\n[3] Diagnosing EDMToFlowWrapper...")
    wrapper = EDMToFlowWrapper(base_model, sigma_max=80.0).to(device)
    
    # Test Point A (Common) and B (Rare)
    pt_a = torch.tensor([[-0.5, -0.5]], device=device, dtype=torch.float32)
    pt_b = torch.tensor([[0.5, 0.5]], device=device, dtype=torch.float32)
    
    # Check Velocity at t=0.5 (Midpoint)
    t_test = 0.5
    v_a = wrapper(pt_a, t_test)
    v_b = wrapper(pt_b, t_test)
    
    print(f"   t={t_test} -> Point A Velocity: {v_a.detach().cpu().numpy()}")
    print(f"   t={t_test} -> Point B Velocity: {v_b.detach().cpu().numpy()}")
    
    if torch.norm(v_a) > 100 or torch.norm(v_b) > 100:
        print("   [ERROR] Velocity is EXPLODING. The Wrapper/Sigma scaling is wrong.")
    else:
        print("   [OK] Velocity magnitudes look reasonable.")

    # 4. Diagnose: The Reward (Surprise)
    # ---------------------------------------------------------
    print("\n[4] Diagnosing Reward Signal (Surprise)...")
    # S-MEME Helper
    smeme = SMEMESolver(base_model, None)
    
    # Calculate Score at Data (t=1 approx)
    # Score = Gradient of Log Likelihood
    # A (High Density) should have small Score (or pointing inward)
    # B (Low Density) should have high Score (or pointing inward strongly? No, score points to mode)
    
    score_a = smeme._get_score_at_data(base_model, pt_a)
    score_b = smeme._get_score_at_data(base_model, pt_b)
    
    print(f"   Score at A (Common): {score_a.detach().cpu().numpy()} | Norm: {torch.norm(score_a):.4f}")
    print(f"   Score at B (Rare):   {score_b.detach().cpu().numpy()} | Norm: {torch.norm(score_b):.4f}")

    # Reward Gradient = -Score
    # We want to check if the Reward Gradient points TOWARDS B for points near B?
    # Actually, "Surprise" usually rewards low density. 
    # If we are at A, -Score points AWAY from A (towards empty space).
    # If we are at B, -Score points AWAY from B (towards empty space).
    # This force pushes particles OUT of the modes.
    
    grad_reward_a = -1.0 * score_a
    print(f"   Reward Grad at A (Direction to move): {grad_reward_a.detach().cpu().numpy()}")
    
    if torch.norm(grad_reward_a) < 1e-4:
        print("   [ERROR] Gradient is VANISHING. No learning signal.")

    # 5. Diagnose: The Adjoint Solve Step
    # ---------------------------------------------------------
    print("\n[5] Diagnosing Solver Step (Does it backprop?)...")
    
    # Mock Solver
    class MockConfig:
        num_inference_steps = 10
        reward_multiplier = 1.0
        per_sample_threshold_quantile = 1.0 # No clipping
        
    solver = AdjointMatchingSolver(wrapper, wrapper, MockConfig())
    
    # Start from Noise
    x_start = torch.randn(10, 2).to(device)
    x_start.requires_grad_(True)
    
    def target_fn(x):
        # maximize x (dummy reward) -> grad is 1
        return torch.ones_like(x) 
        
    # Run Solver
    try:
        loss = solver.solve_and_compute_grad(x_start, None, target_fn, torch.tensor(range(10)))
        print(f"   Solver Forward Loss: {loss.item():.4f}")
        
        # Check Gradient on Weights
        opt.zero_grad()
        loss.backward()
        
        grad_norm = 0.0
        for p in base_model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        
        print(f"   Weight Gradient Norm: {grad_norm:.6f}")
        
        if grad_norm == 0.0:
            print("   [FAIL] Gradients are ZERO. The computational graph is broken (Detached?).")
        elif np.isnan(grad_norm):
             print("   [FAIL] Gradients are NaN. Explosion occurred.")
        else:
            print("   [SUCCESS] Gradients are flowing to the weights.")
            
    except Exception as e:
        print(f"   [CRASH] Solver failed with error: {e}")

if __name__ == "__main__":
    diagnose_smeme_failure()
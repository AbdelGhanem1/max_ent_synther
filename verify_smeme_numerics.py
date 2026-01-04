import torch
import numpy as np
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module
import copy

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_in_region(samples, center, radius=1.0):
    dists = np.linalg.norm(samples - np.array(center), axis=1)
    return np.sum(dists < radius)

def run_numeric_verification():
    device = get_device()
    print(f"Running Numeric Verification on {device}...")
    
    # 1. Setup Bridged Data (Source at 3,3, Target at 0,0)
    n_samples = 10000
    n_a = int(0.95 * n_samples)
    data_a = np.random.normal(loc=[3.0, 3.0], scale=1.5, size=(n_a, 2))
    data_b = np.random.normal(loc=[0.0, 0.0], scale=0.5, size=(n_samples - n_a, 2))
    dataset = torch.from_numpy(np.vstack([data_a, data_b]).astype(np.float32)).to(device)
    
    # 2. Pre-train Base Model
    print("Pre-training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Fast pre-train
    for _ in range(1000):
        indices = torch.randperm(len(dataset))[:256]
        batch = dataset[indices]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # 3. Measure Base Model State
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()
        
    count_base = count_in_region(base_samples, center=[0, 0], radius=1.0)
    print(f"\n[Base Model] Points near Target (0,0): {count_base} / 2000")
    
    if count_base == 0:
        print(">>> CRITICAL FINDING: Base model has ZERO density at target. Exploration is physically impossible.")
    else:
        print(f">>> Bridge Exists: Base model has {count_base} points in target region.")

    # 4. Run S-MEME with Multiplier 0.2 (The "Bridge" candidate)
    print("\nRunning S-MEME (Multiplier 0.2)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.2)
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device) # Frozen previous
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    
    # Define score function
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    # Train 1 Iteration
    opt_smeme = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    # Run 500 steps
    for _ in range(500):
        noise = torch.randn(256, 2).to(device)
        opt_smeme.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_smeme.step()
        
    # 5. Measure S-MEME State
    current_model.eval()
    with torch.no_grad():
        smeme_samples = current_model.sample(batch_size=2000).cpu().numpy()
        
    count_smeme = count_in_region(smeme_samples, center=[0, 0], radius=1.0)
    
    print(f"\n[S-MEME 0.2] Points near Target (0,0): {count_smeme} / 2000")
    print(f"Change: {count_smeme - count_base:+} points")
    
    # 6. Check the "Disappearing" Dots (Multiplier 0.5)
    print("\nChecking Multiplier 0.5 (Explosion Test)...")
    am_config.reward_multiplier = 0.5 # Increase force
    solver.config = am_config # Update solver config
    
    # Run just 100 steps to see trajectory
    for _ in range(100):
        noise = torch.randn(256, 2).to(device)
        opt_smeme.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_smeme.step()
        
    with torch.no_grad():
        exploded_samples = current_model.sample(batch_size=2000).cpu().numpy()
        
    # Statistics
    valid = exploded_samples[np.isfinite(exploded_samples).all(axis=1)]
    print(f"[S-MEME 0.5] Valid Samples: {len(valid)} / 2000")
    if len(valid) > 0:
        print(f"   Min Coord: {valid.min():.2f}")
        print(f"   Max Coord: {valid.max():.2f}")
        if valid.max() > 20 or valid.min() < -20:
            print(">>> CONFIRMED: Model exploded. Particles pushed outside viewable range.")
        else:
            print(">>> STATUS: Model is stable but maybe collapsed.")

if __name__ == "__main__":
    run_numeric_verification()
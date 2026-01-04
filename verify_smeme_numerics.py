import torch
import numpy as np
import copy
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_in_region(samples, center, radius=1.0):
    dists = np.linalg.norm(samples - np.array(center), axis=1)
    return np.sum(dists < radius)

def run_definitive_proof():
    device = get_device()
    print(f"Running Definitive Proof on {device}...")
    
    # 1. SETUP: OVERLAPPING MODES (The "Bridge" is guaranteed)
    # Source: (3,3)
    # Target: (1.5, 1.5) <-- EXACTLY 1 SIGMA AWAY. Impossible to ignore.
    n_samples = 10000
    n_a = int(0.95 * n_samples)
    data_a = np.random.normal(loc=[3.0, 3.0], scale=1.5, size=(n_a, 2))
    data_b = np.random.normal(loc=[1.5, 1.5], scale=0.5, size=(n_samples - n_a, 2))
    dataset = torch.from_numpy(np.vstack([data_a, data_b]).astype(np.float32)).to(device)
    
    # 2. PRE-TRAIN
    print("Pre-training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    for _ in range(1500): # Increased steps slightly to ensure convergence
        indices = torch.randperm(len(dataset))[:256]
        batch = dataset[indices]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # 3. BASELINE METRICS
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()
        
    target_center = [1.5, 1.5]
    count_base = count_in_region(base_samples, center=target_center, radius=0.8)
    print(f"\n[Base Model] Points in Target Region {target_center}: {count_base} / 2000")
    
    if count_base == 0:
        print(">>> FAIL: Base model still has zero density here. Try moving target closer.")
        return

    # 4. RUN S-MEME (Multiplier 0.1 - Paper Standard)
    print("\nRunning S-MEME (Multiplier 0.1)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.1)
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_smeme = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    # Train
    for _ in range(500):
        noise = torch.randn(256, 2).to(device)
        opt_smeme.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_smeme.step()
        
    # 5. S-MEME METRICS
    current_model.eval()
    with torch.no_grad():
        smeme_samples = current_model.sample(batch_size=2000).cpu().numpy()
        
    count_smeme = count_in_region(smeme_samples, center=target_center, radius=0.8)
    
    print(f"\n[S-MEME 0.1] Points in Target Region {target_center}: {count_smeme} / 2000")
    print(f"Change: {count_smeme - count_base:+} points")
    
    # 6. VALIDATE MOVEMENT (Did it expand?)
    dist_base = np.linalg.norm(base_samples - np.array([3.0, 3.0]), axis=1).mean()
    dist_smeme = np.linalg.norm(smeme_samples - np.array([3.0, 3.0]), axis=1).mean()
    print(f"\nAvg Distance from Source Center (3,3):")
    print(f"   Base:  {dist_base:.4f}")
    print(f"   S-MEME:{dist_smeme:.4f}")
    
    if dist_smeme > dist_base:
        print(">>> SUCCESS: S-MEME successfully pushed mass away from the center.")
    else:
        print(">>> FAIL: Model did not expand.")

if __name__ == "__main__":
    run_definitive_proof()
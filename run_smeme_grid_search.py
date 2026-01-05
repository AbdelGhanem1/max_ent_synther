import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# --- MODULE IMPORTS ---
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

# ============================================================================
# 1. SETUP: 70/30 DISTRIBUTION
# ============================================================================
def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    # Mode A: 70% at (-2, -2) - Closer to ensure base model learns it
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_a, 2))
    
    # Mode B: 30% at (2, 2)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[2.0, 2.0], scale=1.0, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def check_balance(samples):
    # Count Cluster A vs Cluster B
    dist_a = np.linalg.norm(samples - np.array([-2.0, -2.0]), axis=1)
    dist_b = np.linalg.norm(samples - np.array([2.0, 2.0]), axis=1)
    
    count_a = np.sum(dist_a < 2.0) 
    count_b = np.sum(dist_b < 2.0)
    
    total = max(len(samples), 1)
    return count_a / total, count_b / total

# ============================================================================
# 2. EXPERIMENT RUNNER
# ============================================================================
def run_k1_debug():
    device = get_device()
    print(">>> 1. Generating Data (70% A / 30% B)...")
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    # --- PRE-TRAIN BASE ---
    print("\n>>> 2. Training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Robust training loop
    for i in range(5000):
        indices = torch.randperm(len(dataset))[:256]
        batch = dataset[indices]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # Verify Base
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()
    ra, rb = check_balance(base_samples)
    print(f"[BASE MODEL] A: {ra*100:.1f}% | B: {rb*100:.1f}%")
    
    if rb < 0.1:
        print("CRITICAL: Base model failed to capture Mode B. Aborting.")
        return

    # --- K=1 SWEEP ---
    alphas = [10.0, 5.0, 2.0, 1.0, 0.5]
    results = {}
    
    print("\n>>> 3. Running K=1 Sweep...")
    
    fig, axes = plt.subplots(1, len(alphas)+1, figsize=(4 * (len(alphas)+1), 4))
    
    # Plot Base
    sns.kdeplot(x=base_samples[:,0], y=base_samples[:,1], fill=True, cmap="Blues", ax=axes[0])
    axes[0].set_title(f"Base\nA:{ra*100:.0f}% B:{rb*100:.0f}%")
    axes[0].set_xlim(-6, 6); axes[0].set_ylim(-6, 6)

    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20)
    
    for i, alpha in enumerate(alphas):
        print(f"   Testing Alpha={alpha} (Multiplier={1.0/alpha:.2f})...")
        
        # Fresh copy of base model for each run
        current_model = copy.deepcopy(base_model)
        prev_model = copy.deepcopy(base_model); prev_model.requires_grad_(False)
        
        am_config.reward_multiplier = 1.0 / alpha
        
        # Setup Solver
        wrapped_curr = DiffusionModelAdapter(current_model, am_config).to(device)
        wrapped_prev = DiffusionModelAdapter(prev_model, am_config).to(device)
        
        solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_prev, model_fine=wrapped_curr, config=am_config)
        smeme_helper = smeme_module.SMEMESolver(wrapped_curr, SMEMEConfig()) # Dummy for helper
        def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_prev, x, t)
        
        # Train (One Iteration)
        opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
        current_model.train()
        
        # Run 500 steps
        for _ in range(500):
            noise = torch.randn(256, 2).to(device)
            opt_s.zero_grad()
            loss = solver.solve_vector_field(noise, entropy_grad)
            loss.backward()
            opt_s.step()
            
        # Eval
        current_model.eval()
        with torch.no_grad():
            samples = current_model.sample(batch_size=2000).cpu().numpy()
            
        r_a, r_b = check_balance(samples)
        results[alpha] = (r_a, r_b)
        print(f"      -> Result: A={r_a*100:.1f}% | B={r_b*100:.1f}%")
        
        # Plot
        ax = axes[i+1]
        valid = samples[np.isfinite(samples).all(axis=1)]
        if len(valid) < 100:
            ax.text(0,0, "Collapsed", ha='center')
        else:
            sns.kdeplot(x=valid[:,0], y=valid[:,1], fill=True, cmap="Oranges", ax=ax)
            ax.set_title(f"Alpha {alpha}\nA:{r_a*100:.0f}% B:{r_b*100:.0f}%")
            ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)

    plt.tight_layout()
    plt.savefig("smeme_k1_sweep.png")
    print("\nSaved 'smeme_k1_sweep.png'")

if __name__ == "__main__":
    run_k1_debug()
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# --- MODULE IMPORTS ---
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

# ============================================================================
# 1. MONKEY PATCH: DEBUGGING & CLAMPING
# ============================================================================
# We patch the solver to (A) Log stats and (B) Use a tighter clamp (5.0 instead of 50.0)
OriginalSolver = smeme_module.VectorFieldAdjointSolver

class DebugSolver(OriginalSolver):
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        # ... (Re-implementing just the critical parts to inject debugs) ...
        # NOTE: We rely on the parent logic for most things, but we need to intercept
        # the 'reward_grad' calculation to clamp and log it.
        
        # To avoid re-writing the whole big function, we wrap the 'vector_field_fn'
        # which computes the gradients.
        
        def debug_wrapper_fn(x, t):
            raw_grad = vector_field_fn(x, t).float()
            
            # 1. Calculate Norms
            norms = torch.norm(raw_grad.reshape(raw_grad.shape[0], -1), dim=1)
            avg_norm = norms.mean().item()
            max_norm = norms.max().item()
            
            # 2. [FIX] Tighter Clamp for Normalized Space (5.0 instead of 50.0)
            # In latent space N(0,1), a force of 50 is massive. 5.0 is safer.
            scale_factor = torch.clamp(norms.view(-1, 1) / 5.0, min=1.0)
            clamped_grad = raw_grad / scale_factor
            
            # 3. Randomly print debug info (1% chance to avoid spam)
            if np.random.rand() < 0.01:
                print(f"   [DEBUG] Score Norms | Avg: {avg_norm:.2f} | Max: {max_norm:.2f} | Clamped: {(clamped_grad.norm(dim=1).mean()):.2f}")
                
            return clamped_grad

        return super().solve_vector_field(x_0, debug_wrapper_fn, prompt_emb)

# Apply the patch
smeme_module.VectorFieldAdjointSolver = DebugSolver

# ============================================================================
# 2. SETUP DATA & MODELS
# ============================================================================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    n_mode_a = int(0.95 * n_samples)
    n_mode_b = n_samples - n_mode_a
    data_a = np.random.normal(loc=[5.0, 5.0], scale=1.5, size=(n_mode_a, 2))
    data_b = np.random.normal(loc=[-5.0, -5.0], scale=1.0, size=(n_mode_b, 2))
    return np.vstack([data_a, data_b]).astype(np.float32)

def plot_density(samples, title, ax, color, limits=((-10, 10), (-10, 10))):
    valid = samples[np.isfinite(samples).all(axis=1)]
    if len(valid) < 100:
        ax.text(0,0, "Model Collapsed", ha='center')
        return
    
    sns.kdeplot(x=valid[:,0], y=valid[:,1], fill=True, cmap=color, levels=10, thresh=0.05, ax=ax)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_title(title, fontsize=10)
    ax.scatter(valid[:500,0], valid[:500,1], s=1, color='black', alpha=0.1)

# ============================================================================
# 3. EXPERIMENT RUNNER
# ============================================================================
def run_experiment(schedule_name, alphas, base_model, raw_data, device):
    print(f"\n>>> RUNNING EXPERIMENT: {schedule_name} | Alphas: {alphas}")
    
    # Clone model to avoid interference
    current_model = copy.deepcopy(base_model)
    
    # Config
    am_config = AdjointMatchingConfig(
        num_train_timesteps=1000, 
        num_inference_steps=20, 
        reward_multiplier=1.0
    )
    smeme_config = SMEMEConfig(
        num_smeme_iterations=len(alphas),
        alpha_schedule=alphas,
        am_config=am_config
    )
    
    # Wrap & Solve
    wrapped = DiffusionModelAdapter(current_model, am_config).to(device)
    solver = smeme_module.SMEMESolver(base_model=wrapped, config=smeme_config)
    # Use safer LR
    solver.optimizer = torch.optim.AdamW(solver.current_model.parameters(), lr=1e-5)
    
    # Train Loop
    batch_size = 256
    def noise_gen():
        while True: yield torch.randn(batch_size, 2).to(device)
        
    # --- FIX START ---
    class Loader:
        def __init__(self, limit=500): 
            self.g = noise_gen()
            self.limit = limit
            self.cnt = 0
            
        def __iter__(self): 
            self.cnt = 0
            return self
            
        def __next__(self): 
            # Stop iteration after 'limit' steps so the solver finishes
            if self.cnt >= self.limit: raise StopIteration
            self.cnt += 1
            return next(self.g)
            
        def __len__(self): return self.limit
        
    # Pass the DATA LOADER, not a range!
    loader = Loader(limit=500) 
    finetuned = solver.train(loader) 
    # --- FIX END ---
    
    # Generate Samples
    print("   Sampling results...")
    finetuned.eval()
    with torch.no_grad():
        samples = finetuned.model.sample(batch_size=2000).cpu().numpy()
        
    return samples
# ============================================================================
# 4. MAIN
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    raw_data = generate_unbalanced_data()
    dataset = torch.from_numpy(raw_data).to(device)
    
    print("1. Pre-training Base Model...")
    base_model = construct_diffusion_model(
        inputs=dataset.cpu(),
        normalizer_type='standard',
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(dataset), batch_size=256, shuffle=True)
    
    base_model.train()
    for _ in tqdm(range(2000), desc="Pre-training"):
        batch = next(iter(loader))[0]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # Get Baseline Samples
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()

    # --- DEFINE GRID ---
    experiments = [
        ("Conservative", (1.0, 0.9, 0.8)),
        ("Balanced",     (1.0, 0.75, 0.5)),
        ("Aggressive",   (1.0, 0.5, 0.1)) # Should work now with 5.0 clamp!
    ]
    
    results = []
    for name, alphas in experiments:
        res = run_experiment(name, alphas, base_model, raw_data, device)
        results.append((name, res))
        
    # --- PLOTTING ---
    print("\nGenerating Grid Search Plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Base
    plot_density(base_samples, "Base Model", axes[0], "Blues")
    axes[0].text(-5, -5, "Target (-5,-5)", color='red', ha='center')
    
    # 2,3,4 Experiments
    cmaps = ["Greens", "Oranges", "Reds"]
    for i, (name, samples) in enumerate(results):
        plot_density(samples, f"S-MEME: {name}", axes[i+1], cmaps[i])
        axes[i+1].text(-5, -5, "Target", color='red', ha='center')
        
    plt.tight_layout()
    plt.savefig("smeme_grid_search.png")
    print("Done. Saved 'smeme_grid_search.png'.")
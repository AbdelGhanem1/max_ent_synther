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
OriginalSolver = smeme_module.VectorFieldAdjointSolver

class DebugSolver(OriginalSolver):
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        def debug_wrapper_fn(x, t):
            raw_grad = vector_field_fn(x, t).float()
            
            # 1. Calculate Norms
            norms = torch.norm(raw_grad.reshape(raw_grad.shape[0], -1), dim=1)
            avg_norm = norms.mean().item()
            max_norm = norms.max().item()
            
            # 2. Safety Clamp (Keep 5.0 for safety, but Paper Schedule shouldn't hit it)
            scale_factor = torch.clamp(norms.view(-1, 1) / 5.0, min=1.0)
            clamped_grad = raw_grad / scale_factor
            
            # 3. Randomly print debug info
            if np.random.rand() < 0.005: 
                print(f"   [DEBUG] Score Norms | Avg: {avg_norm:.2f} | Max: {max_norm:.2f} | Clamped: {(clamped_grad.norm(dim=1).mean()):.2f}")
                
            return clamped_grad

        return super().solve_vector_field(x_0, debug_wrapper_fn, prompt_emb)

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
    # 1. Remove NaNs
    valid = samples[np.isfinite(samples).all(axis=1)]
    
    if len(valid) < 100:
        ax.text(0, 0, "Model Collapsed\n(NaNs)", ha='center')
        ax.set_title(title)
        return

    # 2. Check for Zero Variance
    std_x = np.std(valid[:, 0])
    std_y = np.std(valid[:, 1])
    if std_x < 1e-3 or std_y < 1e-3:
        ax.text(0, 0, "Model Collapsed\n(Zero Variance)", ha='center')
        ax.set_title(title)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        return
    
    try:
        sns.kdeplot(x=valid[:,0], y=valid[:,1], fill=True, cmap=color, levels=10, thresh=0.05, ax=ax)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        ax.set_title(title, fontsize=10)
        ax.scatter(valid[:500,0], valid[:500,1], s=1, color='black', alpha=0.1)
    except Exception as e:
        print(f"Plotting failed for {title}: {e}")
        ax.text(0, 0, "Plot Error", ha='center')

# ============================================================================
# 3. EXPERIMENT RUNNER
# ============================================================================
def run_experiment(schedule_name, alphas, base_model, raw_data, device):
    print(f"\n>>> RUNNING EXPERIMENT: {schedule_name} | Alphas: {alphas}")
    
    current_model = copy.deepcopy(base_model)
    
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
    
    wrapped = DiffusionModelAdapter(current_model, am_config).to(device)
    solver = smeme_module.SMEMESolver(base_model=wrapped, config=smeme_config)
    solver.optimizer = torch.optim.AdamW(solver.current_model.parameters(), lr=1e-5)
    
    batch_size = 256
    def noise_gen():
        while True: yield torch.randn(batch_size, 2).to(device)
        
    class Loader:
        def __init__(self, limit=500): 
            self.g = noise_gen()
            self.limit = limit
            self.cnt = 0
        def __iter__(self): 
            self.cnt = 0
            return self
        def __next__(self): 
            if self.cnt >= self.limit: raise StopIteration
            self.cnt += 1
            return next(self.g)
        def __len__(self): return self.limit
        
    loader = Loader(limit=500) 
    finetuned = solver.train(loader) 
    
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
        
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()

    # --- UPDATED GRID ---
    experiments = [
        # Multipliers: 0.1, 0.1, 0.1 (Paper Baseline)
        ("Paper Fixed", (10.0, 10.0, 10.0)),
        
        # Multipliers: 0.1, 0.2, 0.5 (Gentle Expansion)
        ("Paper Schedule", (10.0, 5.0, 2.0)),
        
        # Multipliers: 1.0, 1.1, 1.25 (Previous "Conservative")
        ("Conservative", (1.0, 0.9, 0.8)) 
    ]
    
    results = []
    for name, alphas in experiments:
        res = run_experiment(name, alphas, base_model, raw_data, device)
        results.append((name, res))
        
    print("\nGenerating Grid Search Plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    plot_density(base_samples, "Base Model", axes[0], "Blues")
    axes[0].text(-5, -5, "Target (-5,-5)", color='red', ha='center')
    
    cmaps = ["Greens", "Oranges", "Reds"]
    for i, (name, samples) in enumerate(results):
        plot_density(samples, f"S-MEME: {name}", axes[i+1], cmaps[i])
        axes[i+1].text(-5, -5, "Target", color='red', ha='center')
        
    plt.tight_layout()
    plt.savefig("smeme_grid_search.png")
    print("Done. Saved 'smeme_grid_search.png'.")
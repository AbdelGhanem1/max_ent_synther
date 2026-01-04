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
# 1. MONKEY PATCH: SAFETY CLAMP
# ============================================================================
OriginalSolver = smeme_module.VectorFieldAdjointSolver

class DebugSolver(OriginalSolver):
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        def debug_wrapper_fn(x, t):
            raw_grad = vector_field_fn(x, t).float()
            norms = torch.norm(raw_grad.reshape(raw_grad.shape[0], -1), dim=1)
            # Gentle clamp to prevent explosions, though Paper Schedule shouldn't hit it
            scale_factor = torch.clamp(norms.view(-1, 1) / 10.0, min=1.0)
            return raw_grad / scale_factor

        return super().solve_vector_field(x_0, debug_wrapper_fn, prompt_emb)

smeme_module.VectorFieldAdjointSolver = DebugSolver

# ============================================================================
# 2. SETUP: THE "BRIDGE" TOPOLOGY
# ============================================================================
def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_bridged_data(n_samples=10000):
    n_a = int(0.95 * n_samples)
    n_b = n_samples - n_a
    
    # Mode A (Common): Large, at (3, 3)
    data_a = np.random.normal(loc=[3.0, 3.0], scale=1.5, size=(n_a, 2))
    
    # Mode B (Rare): Small, at (0, 0) <-- MOVED CLOSER
    # This creates an overlap (Bridge) so the gradient can guide mass here.
    data_b = np.random.normal(loc=[0.0, 0.0], scale=0.5, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def plot_density(samples, title, ax, color, limits=((-5, 10), (-5, 10))):
    valid = samples[np.isfinite(samples).all(axis=1)]
    if len(valid) < 100:
        ax.text(0, 0, "Collapsed", ha='center')
        return
    sns.kdeplot(x=valid[:,0], y=valid[:,1], fill=True, cmap=color, levels=10, thresh=0.05, ax=ax)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_title(title, fontsize=10)
    # Plot reference points
    ax.scatter(0, 0, marker='x', s=100, color='red', label='Target')
    ax.scatter(3, 3, marker='x', s=100, color='blue', label='Source')

# ============================================================================
# 3. SEQUENTIAL EVOLUTION RUNNER
# ============================================================================
def run_final_evolution(alphas, base_model, device):
    print(f"\n>>> RUNNING FINAL PROOF | Alphas: {alphas}")
    
    current_model = copy.deepcopy(base_model)
    history = [] 
    
    # 1. Snapshot Base
    current_model.eval()
    with torch.no_grad():
        history.append(current_model.sample(batch_size=2000).cpu().numpy())

    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20)
    
    prev_model = copy.deepcopy(current_model) 
    prev_model.requires_grad_(False)

    batch_size = 256
    def noise_gen():
        while True: yield torch.randn(batch_size, 2).to(device)
        
    class Loader:
        def __init__(self): self.g = noise_gen(); self.cnt=0
        def __iter__(self): self.cnt=0; return self
        def __next__(self): 
            if self.cnt >= 500: raise StopIteration
            self.cnt+=1; return next(self.g)
        def __len__(self): return 500

    for k, alpha in enumerate(alphas):
        print(f"   Iteration {k+1}/{len(alphas)} (Alpha={alpha}, Multiplier={1.0/alpha:.2f})")
        
        # Configure gentle force
        am_config.reward_multiplier = 1.0 / alpha
        
        # Wrap models
        wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
        wrapped_pre = DiffusionModelAdapter(prev_model, am_config).to(device)
        
        solver = smeme_module.VectorFieldAdjointSolver(
            model_pre=wrapped_pre, 
            model_fine=wrapped_fine,   
            config=am_config
        )
        
        # Score function using Previous Model
        smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
        def entropy_grad(x, t):
            return smeme_helper._get_score_at_data(wrapped_pre, x, t)

        # Train
        opt = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
        loader = Loader()
        
        current_model.train()
        for batch in tqdm(loader, desc=f"Iter {k+1}"):
            opt.zero_grad()
            loss = solver.solve_vector_field(batch, entropy_grad)
            loss.backward()
            opt.step()
            
        # Snapshot
        current_model.eval()
        with torch.no_grad():
            samples = current_model.sample(batch_size=2000).cpu().numpy()
            history.append(samples)
            
        # Update Previous
        prev_model.load_state_dict(current_model.state_dict())
        
    return history

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    
    # 1. Data & Pretraining
    raw_data = generate_bridged_data()
    dataset = torch.from_numpy(raw_data).to(device)
    
    print("Pre-training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(dataset), batch_size=256, shuffle=True)
    
    base_model.train()
    for _ in tqdm(range(2000)):
        batch = next(iter(loader))[0]
        opt.zero_grad(); base_model(batch).backward(); opt.step()

    # 2. Run with PAPER SCHEDULE (Gentle Force)
    # Alpha 10.0 -> Multiplier 0.1 (Paper baseline)
    # Alpha 5.0  -> Multiplier 0.2
    # Alpha 2.0  -> Multiplier 0.5
    alphas = [10.0, 5.0, 2.0]
    
    history = run_final_evolution(alphas, base_model, device)
    
    # 3. Plot
    print("\nGenerating Proof Plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Base Model", "Iter 1 (Multiplier 0.1)", "Iter 2 (Multiplier 0.2)", "Iter 3 (Multiplier 0.5)"]
    
    for i, samples in enumerate(history):
        plot_density(samples, titles[i], axes[i], "Oranges" if i > 0 else "Blues")

    plt.tight_layout()
    plt.savefig("smeme_final_proof.png")
    print("Done. Saved 'smeme_final_proof.png'.")
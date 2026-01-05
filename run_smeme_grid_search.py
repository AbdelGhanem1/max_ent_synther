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
# 1. SETUP: YOUR PROPOSED 70/30 EXPERIMENT
# ============================================================================
def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    # Cluster A: 70% of data (High Density)
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-4.0, -4.0], scale=1.0, size=(n_a, 2))
    
    # Cluster B: 30% of data (Low Density)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[4.0, 4.0], scale=1.0, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def check_balance(samples):
    # Count how many samples fall into Cluster A vs Cluster B
    # We use a simple distance check
    
    dist_a = np.linalg.norm(samples - np.array([-4.0, -4.0]), axis=1)
    dist_b = np.linalg.norm(samples - np.array([4.0, 4.0]), axis=1)
    
    count_a = np.sum(dist_a < 2.5) # Within 2.5 units of center A
    count_b = np.sum(dist_b < 2.5) # Within 2.5 units of center B
    
    total = len(samples)
    ratio_a = count_a / total
    ratio_b = count_b / total
    
    return ratio_a, ratio_b

def plot_results(base_samples, smeme_samples, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    limits = ((-8, 8), (-8, 8))
    
    # Base
    sns.kdeplot(x=base_samples[:,0], y=base_samples[:,1], fill=True, cmap="Blues", ax=axes[0], levels=10)
    axes[0].set_title(f"Base Model\n(Target: 70% A, 30% B)")
    axes[0].set_xlim(limits[0]); axes[0].set_ylim(limits[1])
    # Annotate
    ra, rb = check_balance(base_samples)
    axes[0].text(-4, -4, f"{ra*100:.1f}%", ha='center', color='black', fontweight='bold')
    axes[0].text(4, 4, f"{rb*100:.1f}%", ha='center', color='black', fontweight='bold')

    # S-MEME
    sns.kdeplot(x=smeme_samples[:,0], y=smeme_samples[:,1], fill=True, cmap="Oranges", ax=axes[1], levels=10)
    axes[1].set_title(f"S-MEME Result\n(Expectation: ~50% / 50%)")
    axes[1].set_xlim(limits[0]); axes[1].set_ylim(limits[1])
    # Annotate
    ra_s, rb_s = check_balance(smeme_samples)
    axes[1].text(-4, -4, f"{ra_s*100:.1f}%", ha='center', color='black', fontweight='bold')
    axes[1].text(4, 4, f"{rb_s*100:.1f}%", ha='center', color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"smeme_unbalanced_experiment{title_suffix}.png")
    print(f"Plot saved: smeme_unbalanced_experiment{title_suffix}.png")

# ============================================================================
# 2. EXPERIMENT RUNNER
# ============================================================================
def run_experiment():
    device = get_device()
    print(">>> 1. Generating Data (70% vs 30%)...")
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    print(">>> 2. Training Base Model (Imitator)...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train until it learns the distribution well
    loader = DataLoader(TensorDataset(dataset), batch_size=256, shuffle=True)
    base_model.train()
    for _ in tqdm(range(3000)): 
        batch = next(iter(loader))[0]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # Check Base Stats
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=5000).cpu().numpy()
        
    r_a, r_b = check_balance(base_samples)
    print(f"\n[Base Model Stats] Mode A: {r_a*100:.1f}% | Mode B: {r_b*100:.1f}%")
    
    if r_b < 0.1:
        print("WARNING: Base model failed to learn the rare mode. S-MEME will fail.")
        # But we proceed anyway to see what happens.
        
    print("\n>>> 3. Running S-MEME (Re-Balancer)...")
    # Paper Schedule: start gentle (0.1) then increase
    # alpha 10.0 -> mult 0.1
    # alpha 5.0  -> mult 0.2
    # alpha 2.0  -> mult 0.5
    alphas = [10.0, 5.0, 2.0]
    
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20)
    current_model = copy.deepcopy(base_model)
    prev_model = copy.deepcopy(current_model); prev_model.requires_grad_(False)
    
    batch_size = 256
    def noise_gen():
        while True: yield torch.randn(batch_size, 2).to(device)
    
    class Loader:
        def __init__(self): self.g=noise_gen(); self.cnt=0
        def __iter__(self): self.cnt=0; return self
        def __next__(self):
            if self.cnt >= 500: raise StopIteration
            self.cnt+=1; return next(self.g)
        def __len__(self): return 500

    for k, alpha in enumerate(alphas):
        print(f"   Iteration {k+1} (Alpha {alpha})...")
        am_config.reward_multiplier = 1.0 / alpha
        
        # Wrap
        wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
        wrapped_pre = DiffusionModelAdapter(prev_model, am_config).to(device)
        
        solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
        smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
        def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
        
        opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
        current_model.train()
        loader = Loader()
        
        for batch in tqdm(loader):
            opt_s.zero_grad()
            loss = solver.solve_vector_field(batch, entropy_grad)
            loss.backward()
            opt_s.step()
            
        # Update Previous
        prev_model.load_state_dict(current_model.state_dict())
        
    # Check S-MEME Stats
    current_model.eval()
    with torch.no_grad():
        smeme_samples = current_model.sample(batch_size=5000).cpu().numpy()
        
    rs_a, rs_b = check_balance(smeme_samples)
    print(f"\n[S-MEME Stats] Mode A: {rs_a*100:.1f}% | Mode B: {rs_b*100:.1f}%")
    print(f"Change in Rare Mode B: {rs_b*100 - r_b*100:+.1f}%")
    
    plot_results(base_samples, smeme_samples)

if __name__ == "__main__":
    run_experiment()
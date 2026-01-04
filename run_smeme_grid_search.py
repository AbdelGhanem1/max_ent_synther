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
# 1. SETUP: THE "BRIDGE" TOPOLOGY (Exact same as Numeric Proof)
# ============================================================================
def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_bridged_data(n_samples=10000):
    n_a = int(0.95 * n_samples)
    
    # Source: (3,3)
    data_a = np.random.normal(loc=[3.0, 3.0], scale=1.5, size=(n_a, 2))
    
    # Target: (1.5, 1.5) - The bridge region
    data_b = np.random.normal(loc=[1.5, 1.5], scale=0.5, size=(n_samples - n_a, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def plot_density_comparison(base_samples, smeme_samples):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Common Plot Settings
    limits = ((-2, 8), (-2, 8))
    target_loc = [1.5, 1.5]
    source_loc = [3.0, 3.0]
    
    # --- PLOT 1: BASE MODEL ---
    valid_base = base_samples[np.isfinite(base_samples).all(axis=1)]
    sns.kdeplot(x=valid_base[:,0], y=valid_base[:,1], fill=True, cmap="Blues", levels=10, thresh=0.05, ax=axes[0])
    axes[0].scatter(valid_base[:500,0], valid_base[:500,1], s=2, color='black', alpha=0.1)
    
    # Markers
    axes[0].scatter(*source_loc, marker='x', s=150, color='blue', linewidth=3, label='Source (3,3)')
    axes[0].scatter(*target_loc, marker='o', s=150, facecolors='none', edgecolors='red', linewidth=3, label='Target (1.5,1.5)')
    
    axes[0].set_xlim(limits[0]); axes[0].set_ylim(limits[1])
    axes[0].set_title("Base Model (Before S-MEME)", fontsize=14)
    axes[0].legend(loc='upper right')
    
    # --- PLOT 2: S-MEME ---
    valid_smeme = smeme_samples[np.isfinite(smeme_samples).all(axis=1)]
    sns.kdeplot(x=valid_smeme[:,0], y=valid_smeme[:,1], fill=True, cmap="Oranges", levels=10, thresh=0.05, ax=axes[1])
    axes[1].scatter(valid_smeme[:500,0], valid_smeme[:500,1], s=2, color='black', alpha=0.1)
    
    # Markers
    axes[1].scatter(*source_loc, marker='x', s=150, color='blue', linewidth=3, label='Source')
    axes[1].scatter(*target_loc, marker='o', s=150, facecolors='none', edgecolors='red', linewidth=3, label='Target')
    
    # Add Arrow to show flow
    axes[1].annotate("", xy=(1.5, 1.5), xytext=(3.0, 3.0),
                arrowprops=dict(arrowstyle="->", color='black', lw=3, ls='--'))
    axes[1].text(2.2, 2.2, "Gradient Flow", rotation=45, ha='center', va='center', fontweight='bold')

    axes[1].set_xlim(limits[0]); axes[1].set_ylim(limits[1])
    axes[1].set_title("S-MEME (Multiplier 0.1)", fontsize=14)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("smeme_final_visual_proof.png")
    print("Saved 'smeme_final_visual_proof.png'")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    
    # 1. Data & Pretraining
    raw_data = generate_bridged_data()
    dataset = torch.from_numpy(raw_data).to(device)
    
    print("Pre-training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Fast Pretrain (Same as numeric proof)
    for _ in range(1500):
        indices = torch.randperm(len(dataset))[:256]
        batch = dataset[indices]
        opt.zero_grad(); base_model(batch).backward(); opt.step()
        
    # Snapshot Base
    base_model.eval()
    with torch.no_grad():
        base_samples = base_model.sample(batch_size=2000).cpu().numpy()

    # 2. Run S-MEME
    print("Running S-MEME...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.1)
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_smeme = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    for _ in tqdm(range(500)):
        noise = torch.randn(256, 2).to(device)
        opt_smeme.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_smeme.step()
        
    # Snapshot S-MEME
    current_model.eval()
    with torch.no_grad():
        smeme_samples = current_model.sample(batch_size=2000).cpu().numpy()
        
    # 3. Plot
    plot_density_comparison(base_samples, smeme_samples)
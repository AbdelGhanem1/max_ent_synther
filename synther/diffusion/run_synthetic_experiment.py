import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# --- YOUR ACTUAL MODULES ---
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.diffusion_config import edm_global_config
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
from synther.diffusion.smeme_solver import SMEMESolver

# ============================================================================
# 1. SETUP & SYNTHETIC DATA GENERATION
# ============================================================================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    """
    Creates the 'Unbalanced Modes' dataset:
    - Mode A (High Density): 95% of data at (5, 5)
    - Mode B (Low Density):   5% of data at (-5, -5)
    """
    n_mode_a = int(0.95 * n_samples)
    n_mode_b = n_samples - n_mode_a
    
    # Mode A: Large and heavy
    data_a = np.random.normal(loc=[5.0, 5.0], scale=1.5, size=(n_mode_a, 2))
    
    # Mode B: Small and rare (The "Discovery" target)
    data_b = np.random.normal(loc=[-5.0, -5.0], scale=1.0, size=(n_mode_b, 2))
    
    data = np.vstack([data_a, data_b])
    np.random.shuffle(data)
    return data.astype(np.float32)

def plot_density(samples, title, ax, color, limits=((-10, 10), (-10, 10))):
    sns.kdeplot(
        x=samples[:, 0], y=samples[:, 1], 
        fill=True, cmap=color, levels=15, thresh=0.05, ax=ax
    )
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_title(title, fontsize=14)
    ax.scatter(samples[:500, 0], samples[:500, 1], s=2, color='black', alpha=0.1)

# ============================================================================
# 2. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(f"Running S-MEME Synthetic Test on {device}...")
    
    # --- A. DATA PREPARATION ---
    raw_data = generate_unbalanced_data()
    dataset_tensor = torch.from_numpy(raw_data).to(device)
    
    # --- B. CONSTRUCT BASE MODEL (Using your Utils) ---
    print("Constructing Base Model...")
    # Note: We pass the raw data so utils.py can calculate Mean/Std for normalization
    base_model = construct_diffusion_model(
        inputs=dataset_tensor.cpu(),
        normalizer_type='standard', # Standard Gaussian Normalization
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    # --- C. PRE-TRAIN BASE MODEL (Simulate "Offline RL" pre-training) ---
    print("Pre-training Base Model (The 'Imitator')...")
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    batch_size = 256
    n_steps = 2000 # Fast training for synthetic data
    
    loader = DataLoader(TensorDataset(dataset_tensor), batch_size=batch_size, shuffle=True)
    loader_iter = iter(loader)
    
    base_model.train()
    for step in tqdm(range(n_steps), desc="Pre-training"):
        try:
            batch = next(loader_iter)[0]
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)[0]
            
        loss = base_model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Pre-training Complete.")
    
    # --- D. S-MEME CONFIGURATION ---
    # We want to be aggressive to visualize the shift
    am_config = AdjointMatchingConfig(
        num_train_timesteps=1000, 
        num_inference_steps=20,  # Fast solving
        reward_multiplier=1.0    # Will be overridden by alpha
    )
    
    smeme_config = SMEMEConfig(
        num_smeme_iterations=3,
        alpha_schedule=(1.0, 0.5, 0.1), # Decreasing alpha = Increasing Exploration
        am_config=am_config
    )
    
    # Wrap model with the Adapter (Using the Config Source of Truth)
    wrapped_model = DiffusionModelAdapter(base_model, am_config).to(device)
    
    # Initialize Solver
    solver = SMEMESolver(base_model=wrapped_model, config=smeme_config)
    
    # --- E. S-MEME FINE-TUNING ---
    # Generator for S-MEME (it needs infinite noise source)
    def noise_generator():
        while True:
            # We generate 2D noise
            yield torch.randn(batch_size, 2).to(device)
            
    # S-MEME Loop
    # We use fewer steps than real RL because 2D converges fast
    steps_per_iter = 500 
    
    class SimpleLoader:
        def __init__(self, gen, limit):
            self.gen = gen; self.limit = limit
        def __iter__(self): self.cnt = 0; return self
        def __next__(self):
            if self.cnt >= self.limit: raise StopIteration
            self.cnt += 1
            return next(self.gen)

    train_loader = SimpleLoader(noise_generator(), steps_per_iter)
    
    print("\nStarting S-MEME Fine-tuning...")
    finetuned_wrapper = solver.train(train_loader)
    finetuned_model = finetuned_wrapper.model
    
    # --- F. VISUALIZATION ---
    print("\nGenerating Visualization...")
    base_model.eval()
    finetuned_model.eval()
    
    # Generate samples
    n_plot = 2000
    with torch.no_grad():
        # Note: .sample() automatically un-normalizes using the stats from Step B
        samples_base = base_model.sample(batch_size=n_plot).cpu().numpy()
        samples_smeme = finetuned_model.sample(batch_size=n_plot).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Ground Truth
    plot_density(raw_data, "Ground Truth (Unbalanced)", axes[0], "Greens")
    axes[0].text(-5, -3, "Rare Mode\n(5%)", color='red', ha='center', fontweight='bold')
    axes[0].text(5, 3, "Common Mode\n(95%)", color='blue', ha='center', fontweight='bold')

    # 2. Base Model
    plot_density(samples_base, "Base Model (Imitator)", axes[1], "Blues")
    # Calculate ratio roughly
    n_rare_base = np.sum(samples_base[:, 0] < 0)
    axes[1].text(-5, -8, f"Count: {n_rare_base}", color='red', ha='center')

    # 3. S-MEME
    plot_density(samples_smeme, "S-MEME (Explorer)", axes[2], "Oranges")
    n_rare_smeme = np.sum(samples_smeme[:, 0] < 0)
    axes[2].text(-5, -8, f"Count: {n_rare_smeme}", color='red', ha='center')
    
    plt.tight_layout()
    plt.savefig("smeme_synthetic_test.png")
    print(f"\nTest Complete. Saved 'smeme_synthetic_test.png'.")
    print(f"Rare Mode Count: Base={n_rare_base} vs S-MEME={n_rare_smeme}")
    
    if n_rare_smeme > n_rare_base * 1.5:
        print("✅ SUCCESS: S-MEME significantly increased exploration of the rare mode.")
    else:
        print("⚠️ WARNING: Exploration increase was minimal. Check gradients.")
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
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
    # 1. Remove NaNs and Infs
    valid_mask = np.isfinite(samples).all(axis=1)
    clean_samples = samples[valid_mask]
    
    if len(clean_samples) < 100:
        ax.text(0, 0, "Model Collapsed\n(NaNs or bad data)", ha='center')
        ax.set_title(title)
        return

    # 2. Check for variance (prevents "Contour levels must be increasing" crash)
    if np.std(clean_samples) < 1e-3:
        ax.text(0, 0, "Model Collapsed\n(Zero Variance)", ha='center')
        ax.set_title(title)
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        return

    try:
        sns.kdeplot(
            x=clean_samples[:, 0], y=clean_samples[:, 1], 
            fill=True, cmap=color, levels=15, thresh=0.05, ax=ax
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_title(title, fontsize=14)
        # Plot raw points for verification
        subset = clean_samples[:500]
        ax.scatter(subset[:, 0], subset[:, 1], s=2, color='black', alpha=0.1)
    except Exception as e:
        print(f"Warning: Plotting failed for {title}: {e}")
        ax.text(0, 0, "Plotting Error", ha='center')

# ============================================================================
# 2. MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true', help="Run minimal steps to check for bugs")
    args = parser.parse_args()

    device = get_device()
    print(f"Running S-MEME Synthetic Test on {device} (Dry Run: {args.dry_run})")
    
    # --- A. DATA PREPARATION ---
    raw_data = generate_unbalanced_data()
    dataset_tensor = torch.from_numpy(raw_data).to(device)
    
    # --- B. CONSTRUCT BASE MODEL ---
    print("Constructing Base Model...")
    base_model = construct_diffusion_model(
        inputs=dataset_tensor.cpu(),
        normalizer_type='standard',
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    # --- C. PRE-TRAIN BASE MODEL ---
    # Fast training for synthetic data
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    batch_size = 256
    n_pretrain_steps = 100 if args.dry_run else 2000
    
    print(f"Pre-training Base Model for {n_pretrain_steps} steps...")
    loader = DataLoader(TensorDataset(dataset_tensor), batch_size=batch_size, shuffle=True)
    loader_iter = iter(loader)
    
    base_model.train()
    for step in tqdm(range(n_pretrain_steps), desc="Pre-training"):
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
    
    # [CRITICAL] Save a copy of the base model BEFORE S-MEME modifies it in-place
    original_base_model = copy.deepcopy(base_model)
    
    # --- D. S-MEME CONFIGURATION ---
    am_config = AdjointMatchingConfig(
        num_train_timesteps=1000, 
        num_inference_steps=20,
        reward_multiplier=1.0 
    )
    
    # In Dry Run, we do 1 iteration. In Full Run, we do 3.
    smeme_iterations = 1 if args.dry_run else 3
    alpha_schedule = (1.0,) if args.dry_run else (1.0, 0.5, 0.1)

    smeme_config = SMEMEConfig(
        num_smeme_iterations=smeme_iterations,
        alpha_schedule=alpha_schedule,
        am_config=am_config
    )
    
    # Wrap model with the Adapter
    wrapped_model = DiffusionModelAdapter(base_model, am_config).to(device)
    
    # Initialize Solver
    solver = SMEMESolver(base_model=wrapped_model, config=smeme_config)
    
    # [FIX] Override the solver's internal optimizer with a lower LR for synthetic data
    # The default 1e-4 is often too aggressive for this delicate 2D manifold
    solver.optimizer = torch.optim.AdamW(solver.current_model.parameters(), lr=1e-5, weight_decay=1e-2)
    
    # --- E. S-MEME FINE-TUNING ---
    def noise_generator():
        while True:
            yield torch.randn(batch_size, 2).to(device)
            
    # S-MEME Loop
    steps_per_iter = 10 if args.dry_run else 500
    
    class SimpleLoader:
        def __init__(self, gen, limit):
            self.gen = gen; self.limit = limit
        def __iter__(self): self.cnt = 0; return self
        def __next__(self):
            if self.cnt >= self.limit: raise StopIteration
            self.cnt += 1
            return next(self.gen)

    train_loader = SimpleLoader(noise_generator(), steps_per_iter)
    
    print(f"\nStarting S-MEME Fine-tuning ({steps_per_iter} steps per iter)...")
    finetuned_wrapper = solver.train(train_loader)
    finetuned_model = finetuned_wrapper.model
    
    # --- F. VISUALIZATION ---
    print("\nGenerating Visualization...")
    original_base_model.eval()
    finetuned_model.eval()
    
    n_plot = 2000
    with torch.no_grad():
        # Generate from Original
        samples_base = original_base_model.sample(batch_size=n_plot).cpu().numpy()
        # Generate from S-MEME
        samples_smeme = finetuned_model.sample(batch_size=n_plot).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Ground Truth
    plot_density(raw_data, "Ground Truth (Unbalanced)", axes[0], "Greens")
    axes[0].text(-5, -3, "Rare Mode\n(5%)", color='red', ha='center', fontweight='bold')
    axes[0].text(5, 3, "Common Mode\n(95%)", color='blue', ha='center', fontweight='bold')

    # 2. Base Model
    plot_density(samples_base, "Base Model (Imitator)", axes[1], "Blues")
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
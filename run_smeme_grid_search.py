import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# --- IMPORTS ---
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_safe_data(n_samples=10000):
    # SCALED DOWN DATA (Fits safely inside [-1, 1])
    # Cluster A: 70% at (-0.5, -0.5)
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-0.5, -0.5], scale=0.15, size=(n_a, 2))
    
    # Cluster B: 30% at (0.5, 0.5)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[0.5, 0.5], scale=0.15, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def check_counts(samples):
    # Count points near A (-0.5) vs B (0.5)
    # Using small radius 0.4
    da = np.linalg.norm(samples - np.array([-0.5, -0.5]), axis=1)
    db = np.linalg.norm(samples - np.array([0.5, 0.5]), axis=1)
    
    ca = np.sum(da < 0.4)
    cb = np.sum(db < 0.4)
    total = len(samples)
    return ca, cb, total

def run_safe_experiment():
    device = get_device()
    print(">>> 1. Generating Scaled Data (70% @ -0.5, 30% @ 0.5)...")
    dataset = torch.from_numpy(generate_safe_data()).to(device)
    
    # --- TRAIN BASE ---
    print(">>> 2. Training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train robustly
    for _ in tqdm(range(3000)):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()
        
    # Check Base
    base_model.eval()
    with torch.no_grad():
        base_samps = base_model.sample(batch_size=2000).cpu().numpy()
    ca, cb, tot = check_counts(base_samps)
    print(f"\n[BASE MODEL] A: {ca/tot*100:.1f}% | B: {cb/tot*100:.1f}%")

    if cb/tot < 0.1:
        print("WARNING: Base model missed the rare mode again. Re-training won't help if random seed is unlucky.")
    
    # --- RUN S-MEME ---
    # We use Alpha=2.0 (Multiplier 0.5) to verify rebalancing
    print("\n>>> 3. Running S-MEME (Alpha=2.0)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.5)
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    for _ in tqdm(range(1000)): # 1000 steps to allow movement
        noise = torch.randn(256, 2).to(device)
        opt_s.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_s.step()
        
    # --- RESULTS ---
    current_model.eval()
    with torch.no_grad():
        smeme_samps = current_model.sample(batch_size=2000).cpu().numpy()
        
    csa, csb, stot = check_counts(smeme_samps)
    print(f"\n[S-MEME] A: {csa/stot*100:.1f}% | B: {csb/stot*100:.1f}%")
    
    # PLOT
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Base Plot
    sns.kdeplot(x=base_samps[:,0], y=base_samps[:,1], fill=True, cmap="Blues", ax=axes[0])
    axes[0].scatter(base_samps[:300,0], base_samps[:300,1], s=5, color='black', alpha=0.3)
    axes[0].set_title(f"Base Model\nA: {ca/tot*100:.0f}%")
    axes[0].set_xlim(-1.5, 1.5); axes[0].set_ylim(-1.5, 1.5)
    
    # S-MEME Plot
    sns.kdeplot(x=smeme_samps[:,0], y=smeme_samps[:,1], fill=True, cmap="Oranges", ax=axes[1])
    axes[1].scatter(smeme_samps[:300,0], smeme_samps[:300,1], s=5, color='black', alpha=0.3)
    axes[1].set_title(f"S-MEME (Alpha 2.0)\nA: {csa/stot*100:.0f}% (Target: Lower)")
    axes[1].set_xlim(-1.5, 1.5); axes[1].set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig("smeme_safe_experiment.png")
    print("\nSaved 'smeme_safe_experiment.png'")

if __name__ == "__main__":
    run_safe_experiment()
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, TensorDataset

# --- IMPORTS ---
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_a, 2))
    data_b = np.random.normal(loc=[2.0, 2.0], scale=1.0, size=(n_samples - n_a, 2))
    return np.vstack([data_a, data_b]).astype(np.float32)

def run_scatter_analysis():
    device = get_device()
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    # 1. Train Base (Fast)
    print("Training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    for _ in range(3000):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()
        
    # 2. Run S-MEME (Alpha 2.0 - Best Balance from sweep)
    print("Running S-MEME (Alpha 2.0)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.5) # 1/2.0 = 0.5
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    for _ in range(500):
        noise = torch.randn(256, 2).to(device)
        opt_s.zero_grad(); solver.solve_vector_field(noise, entropy_grad).backward(); opt_s.step()
        
    # 3. Generate Scatter Plots
    base_model.eval(); current_model.eval()
    with torch.no_grad():
        samples_base = base_model.sample(batch_size=2000).cpu().numpy()
        samples_smeme = current_model.sample(batch_size=2000).cpu().numpy()
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot Base
    axes[0].scatter(samples_base[:,0], samples_base[:,1], s=10, alpha=0.5, label='Samples')
    axes[0].set_title("Base Model")
    axes[0].set_xlim(-6, 6); axes[0].set_ylim(-6, 6)
    # Draw Circles for Count Zones
    circle_a = plt.Circle((-2, -2), 2.0, color='r', fill=False, lw=2, label='Cluster A Zone')
    circle_b = plt.Circle((2, 2), 2.0, color='g', fill=False, lw=2, label='Cluster B Zone')
    axes[0].add_patch(circle_a); axes[0].add_patch(circle_b)
    
    # Plot S-MEME
    axes[1].scatter(samples_smeme[:,0], samples_smeme[:,1], s=10, alpha=0.5, color='orange')
    axes[1].set_title("S-MEME (Alpha 2.0)")
    axes[1].set_xlim(-6, 6); axes[1].set_ylim(-6, 6)
    circle_a2 = plt.Circle((-2, -2), 2.0, color='r', fill=False, lw=2)
    circle_b2 = plt.Circle((2, 2), 2.0, color='g', fill=False, lw=2)
    axes[1].add_patch(circle_a2); axes[1].add_patch(circle_b2)
    
    plt.tight_layout()
    plt.savefig("smeme_scatter_analysis.png")
    print("Saved 'smeme_scatter_analysis.png'")

if __name__ == "__main__":
    run_scatter_analysis()
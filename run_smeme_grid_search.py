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
    # Cluster A: 70% at (-2, -2)
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_a, 2))
    # Cluster B: 30% at (2, 2)
    data_b = np.random.normal(loc=[2.0, 2.0], scale=1.0, size=(n_samples - n_a, 2))
    return np.vstack([data_a, data_b]).astype(np.float32)

def run_time_evolution():
    device = get_device()
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    print(">>> 1. Training Robust Base Model (70/30)...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train robustly (4000 steps)
    for _ in range(4000):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()

    # --- S-MEME TIME EVOLUTION ---
    # We use Alpha = 1.0 (Multiplier 1.0) - Strong enough to move, safe enough to track
    print("\n>>> 2. Running S-MEME Time Evolution (Alpha 1.0)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=1.0) 
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    # Snapshots
    checkpoints = [0, 500, 1000, 2000]
    history = []
    
    step = 0
    while step <= 2000:
        if step in checkpoints:
            print(f"   Snapshot at step {step}...")
            current_model.eval()
            with torch.no_grad():
                samples = current_model.sample(batch_size=2000).cpu().numpy()
            history.append((step, samples))
            current_model.train()
            
        if step == 2000: break
        
        # Training Step
        noise = torch.randn(256, 2).to(device)
        opt_s.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_s.step()
        step += 1

    # --- PLOTTING ---
    print("\nGenerating Evolution Plots...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (stp, samps) in enumerate(history):
        ax = axes[i]
        # Scatter for precision
        ax.scatter(samps[:,0], samps[:,1], s=5, alpha=0.4, color='orange' if i>0 else 'blue')
        
        # Draw Zones
        ax.add_patch(plt.Circle((-2, -2), 2.0, color='r', fill=False, lw=2, label='A (Start)'))
        ax.add_patch(plt.Circle((2, 2), 2.0, color='g', fill=False, lw=2, label='B (Goal)'))
        
        # Calculate Balance
        da = np.linalg.norm(samps - np.array([-2,-2]), axis=1)
        db = np.linalg.norm(samps - np.array([2,2]), axis=1)
        ca = np.sum(da < 2.0)
        cb = np.sum(db < 2.0)
        total = len(samps)
        
        ax.set_title(f"Step {stp}\nA:{ca/total*100:.0f}%  B:{cb/total*100:.0f}%")
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        if i==0: ax.legend()

    plt.tight_layout()
    plt.savefig("smeme_time_evolution.png")
    print("Saved 'smeme_time_evolution.png'")

if __name__ == "__main__":
    run_time_evolution()
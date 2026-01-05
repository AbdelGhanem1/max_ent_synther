import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_scaled_data(n_samples=10000):
    # Cluster A: 70% at (-0.5, -0.5)
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-0.5, -0.5], scale=0.15, size=(n_a, 2))
    
    # Cluster B: 30% at (0.5, 0.5)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[0.5, 0.5], scale=0.15, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def run_time_evolution_scaled():
    device = get_device()
    dataset = torch.from_numpy(generate_scaled_data()).to(device)
    
    print(">>> 1. Training Base Model (70/30 at Safe Scale)...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train robustly (4000 steps)
    for _ in range(4000):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()

    # --- S-MEME TIME EVOLUTION ---
    # Using Alpha = 2.0 (Multiplier 0.5) to encourage movement
    print("\n>>> 2. Running S-MEME Time Evolution (Alpha 2.0)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=0.5) 
    
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
    # Increase batch size for plot clarity
    plot_batch_size = 3000
    
    while step <= 2000:
        if step in checkpoints:
            print(f"   Snapshot at step {step}...")
            current_model.eval()
            with torch.no_grad():
                samples = current_model.sample(batch_size=plot_batch_size).cpu().numpy()
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
    print("\nGenerating Scaled Evolution Plots...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (stp, samps) in enumerate(history):
        ax = axes[i]
        
        # Scatter for precision (Orange for S-MEME)
        color = 'blue' if i == 0 else 'orange'
        ax.scatter(samps[:,0], samps[:,1], s=5, alpha=0.3, color=color)
        
        # Draw Zones (Scaled Down)
        # Cluster A (-0.5)
        ax.add_patch(plt.Circle((-0.5, -0.5), 0.4, color='r', fill=False, lw=2, label='A (Start)'))
        # Cluster B (0.5)
        ax.add_patch(plt.Circle((0.5, 0.5), 0.4, color='g', fill=False, lw=2, label='B (Goal)'))
        
        # Calculate Balance
        da = np.linalg.norm(samps - np.array([-0.5,-0.5]), axis=1)
        db = np.linalg.norm(samps - np.array([0.5,0.5]), axis=1)
        ca = np.sum(da < 0.4)
        cb = np.sum(db < 0.4)
        total = len(samps)
        
        ax.set_title(f"Step {stp}\nA:{ca/total*100:.0f}%  B:{cb/total*100:.0f}%")
        
        # Set limits to [-1.5, 1.5] so we see movement but stay zoomed in
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        
        if i==0: ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("smeme_time_evolution_scaled.png")
    print("Saved 'smeme_time_evolution_scaled.png'")

if __name__ == "__main__":
    run_time_evolution_scaled()
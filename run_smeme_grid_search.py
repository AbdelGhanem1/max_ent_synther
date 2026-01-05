import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=5000):
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-2.0, -2.0], scale=1.0, size=(n_a, 2))
    data_b = np.random.normal(loc=[2.0, 2.0], scale=1.0, size=(n_samples - n_a, 2))
    return np.vstack([data_a, data_b]).astype(np.float32)

def diagnose_collapse():
    device = get_device()
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    print(">>> 1. Training Base Model...")
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    # Fast robust training
    for _ in range(3000):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()

    print("\n>>> 2. Running S-MEME Diagnostics (Alpha=1.0)...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=1.0)
    
    current_model = copy.deepcopy(base_model)
    wrapped_fine = DiffusionModelAdapter(current_model, am_config).to(device)
    wrapped_pre = DiffusionModelAdapter(base_model, am_config).to(device)
    
    solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    smeme_helper = smeme_module.SMEMESolver(wrapped_fine, SMEMEConfig())
    def entropy_grad(x, t): return smeme_helper._get_score_at_data(wrapped_pre, x, t)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    current_model.train()
    
    # Track statistics
    checkpoints = [0, 500, 1000, 2000]
    history = []

    print(f"{'Step':<6} | {'Mean (X, Y)':<20} | {'Std (X, Y)':<20} | {'Min / Max':<20} | {'Sample Point'}")
    print("-" * 90)

    for step in range(2001):
        if step in checkpoints:
            current_model.eval()
            with torch.no_grad():
                samples = current_model.sample(batch_size=2000).cpu().numpy()
            
            # CALC STATS
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
            min_val = samples.min()
            max_val = samples.max()
            sample_pt = samples[0] # Just pick the first one
            
            print(f"{step:<6} | {mean[0]:.2f}, {mean[1]:.2f}{'':<8} | {std[0]:.2f}, {std[1]:.2f}{'':<8} | {min_val:.1f} / {max_val:.1f}{'':<6} | {sample_pt}")
            history.append((step, samples))
            current_model.train()

        # Step
        noise = torch.randn(256, 2).to(device)
        opt_s.zero_grad()
        loss = solver.solve_vector_field(noise, entropy_grad)
        loss.backward()
        opt_s.step()

    # --- PLOT ---
    print("\nGenerating Diagnostic Plot...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (stp, samps) in enumerate(history):
        ax = axes[i]
        
        # Determine plot limits dynamically to catch "drifting" points
        xmax, ymax = np.abs(samps).max(axis=0)
        limit = max(6, xmax + 2, ymax + 2) # Expand if points are far out
        
        ax.scatter(samps[:,0], samps[:,1], s=10, alpha=0.5, color='orange' if i>0 else 'blue')
        
        # Reference Zones
        ax.add_patch(plt.Circle((-2, -2), 2.0, color='r', fill=False, lw=2))
        ax.add_patch(plt.Circle((2, 2), 2.0, color='g', fill=False, lw=2))
        
        ax.set_title(f"Step {stp}\nStd: {samps.std():.2f}")
        ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit) # Dynamic limits

    plt.tight_layout()
    plt.savefig("smeme_diagnostic.png")
    print("Saved 'smeme_diagnostic.png'")

if __name__ == "__main__":
    diagnose_collapse()
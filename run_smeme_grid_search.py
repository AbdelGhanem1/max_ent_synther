import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

# Imports from your codebase (Assuming you updated the files as discussed)
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from synther.diffusion.smeme_solver import SMEMESolver, EDMToFlowWrapper
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver

def get_device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_scaled_data(n_samples=10000):
    # Cluster A: 90% at (-0.5, -0.5) (High Density)
    n_a = int(0.90 * n_samples)
    data_a = np.random.normal(loc=[-0.5, -0.5], scale=0.10, size=(n_a, 2))
    
    # Cluster B: 10% at (0.5, 0.5) (Rare Mode)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[0.5, 0.5], scale=0.10, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def run_time_evolution_scaled():
    device = get_device()
    dataset = torch.from_numpy(generate_scaled_data()).to(device)
    
    print(">>> 1. Training Base Model (90% A / 10% B)...")
    # Base model training (Standard EDM)
    base_model = construct_diffusion_model(inputs=dataset.cpu(), normalizer_type='standard', denoising_network=ResidualMLPDenoiser).to(device)
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    
    # Train enough to learn the imbalance
    for i in range(2000):
        indices = torch.randperm(len(dataset))[:256]
        loss = base_model(dataset[indices]) # EDM forward returns loss
        opt.zero_grad(); loss.backward(); opt.step()

    # --- S-MEME TIME EVOLUTION ---
    print("\n>>> 2. Running S-MEME Iteration 1 (Expect Inversion)...")
    
    # Config: High multiplier to force exploration quickly for the plot
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20, reward_multiplier=1.0) 
    
    current_model = copy.deepcopy(base_model)
    
    # --- CHANGE 1: Use the new Wrappers ---
    wrapped_fine = EDMToFlowWrapper(current_model).to(device)
    wrapped_pre = EDMToFlowWrapper(base_model).to(device)
    
    # --- CHANGE 2: Use the new Solver ---
    solver = AdjointMatchingSolver(model_pre=wrapped_pre, model_fine=wrapped_fine, config=am_config)
    
    # Helper to calculate the "Surprise" Gradient (-Score)
    # We use the helper method from the SMEMESolver class logic
    temp_smeme = SMEMESolver(base_model, SMEMEConfig()) # Just to access the static helper method
    
    def surprise_grad_fn(x):
        # Returns -Score (Points towards low density)
        score = temp_smeme._get_score_at_data(base_model, x)
        return -1.0 * score 

    opt_s = torch.optim.AdamW(current_model.parameters(), lr=5e-5) # Low LR for fine-tuning
    current_model.train()
    
    # Snapshots
    checkpoints = [0, 500, 1000, 2000]
    history = []
    
    step = 0
    plot_batch_size = 2000
    
    while step <= 2000:
        # Snapshot Logic
        if step in checkpoints:
            print(f"   Snapshot at step {step}...")
            current_model.eval()
            with torch.no_grad():
                # Sample using the standard EDM sampler (not the flow wrapper)
                # because we want to see what the actual model produces
                samples = current_model.sample(batch_size=plot_batch_size).cpu().numpy()
            history.append((step, samples))
            current_model.train()
        
        if step == 2000: break
        
        # Training Step
        # S-MEME starts from Pure Noise (standard Gaussian)
        noise = torch.randn(256, 2).to(device)
        
        opt_s.zero_grad()
        
        # --- CHANGE 3: New Call Signature ---
        loss = solver.solve_and_compute_grad(
            x_start=noise, 
            prompt_emb=None, 
            target_grad_fn=surprise_grad_fn,
            active_indices=torch.tensor(range(20))
        )
        
        # Backward is already done in solve_and_compute_grad? 
        # Wait, my previous code returned a differentiable loss tensor.
        loss.backward()
        
        opt_s.step()
        step += 1

    # --- PLOTTING ---
    print("\nGenerating Plots...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (stp, samps) in enumerate(history):
        ax = axes[i]
        
        # Draw Zones
        ax.add_patch(plt.Circle((-0.5, -0.5), 0.3, color='r', fill=False, lw=2, label='A (Common)'))
        ax.add_patch(plt.Circle((0.5, 0.5), 0.3, color='g', fill=False, lw=2, label='B (Rare)'))
        
        # Scatter
        color = 'blue' if i == 0 else 'orange'
        ax.scatter(samps[:,0], samps[:,1], s=5, alpha=0.3, color=color)
        
        # Calculate Balance
        da = np.linalg.norm(samps - np.array([-0.5,-0.5]), axis=1)
        db = np.linalg.norm(samps - np.array([0.5,0.5]), axis=1)
        ca = np.sum(da < 0.3)
        cb = np.sum(db < 0.3)
        total = len(samps)
        
        ax.set_title(f"Step {stp}\nA:{ca/total*100:.0f}%  B:{cb/total*100:.0f}%")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        if i==0: ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("smeme_inversion_test.png")
    print("Saved 'smeme_inversion_test.png'")

if __name__ == "__main__":
    run_time_evolution_scaled()
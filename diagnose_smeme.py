import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from torch.utils.data import DataLoader, TensorDataset

# Import your modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig, DiffusionModelAdapter
import synther.diffusion.smeme_solver as smeme_module # Import the module to patch it

# ============================================================================
# 1. DIAGNOSTIC TOOLS
# ============================================================================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_unbalanced_data(n_samples=10000):
    # Same data generation as before
    n_mode_a = int(0.95 * n_samples)
    n_mode_b = n_samples - n_mode_a
    data_a = np.random.normal(loc=[5.0, 5.0], scale=1.5, size=(n_mode_a, 2))
    data_b = np.random.normal(loc=[-5.0, -5.0], scale=1.0, size=(n_mode_b, 2))
    data = np.vstack([data_a, data_b])
    np.random.shuffle(data)
    return data.astype(np.float32)

def plot_vector_field(model, device, title, ax):
    """
    Visualizes the Entropy Force (Score Function) of the model.
    This shows us WHERE S-MEME wants to push the data.
    """
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(x, y)
    
    # Grid points
    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    points_tensor = torch.from_numpy(points).float().to(device)
    
    # We query the score at t=0 (Data distribution)
    # Note: In our Adapter, t_idx=0 means CLEAN DATA (Low Noise) in the Solver sense?
    # Let's check Adapter: t=1000 -> Sigma=80, t=0 -> Sigma=0.002.
    # We want to see the score at the data level.
    t_idx = torch.zeros(len(points), device=device).long() 
    
    with torch.no_grad():
        # Get Epsilon prediction
        eps = model(points_tensor, t_idx)
        # Convert to Score: s = -eps / sigma
        # We know sigma at t=0 is 0.002
        score = -eps / 0.002
        
    U = score[:, 0].cpu().numpy().reshape(X.shape)
    V = score[:, 1].cpu().numpy().reshape(X.shape)
    
    # Normalize lengths for cleaner visualization
    M = np.hypot(U, V)
    M[M == 0] = 1
    U /= M
    V /= M
    
    ax.quiver(X, Y, U, V, M, cmap='coolwarm')
    ax.set_title(title)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

# ============================================================================
# 2. MONKEY PATCH FOR LOGGING
# ============================================================================
# We define a custom solver that logs the "Reward Magnitude" inside the loop
OriginalSolver = smeme_module.VectorFieldAdjointSolver

class DiagnosticSolver(OriginalSolver):
    def solve_vector_field(self, x_0, vector_field_fn, prompt_emb=None):
        # We wrap the original method but adding prints is hard without copy-paste.
        # Instead, we intercept the 'vector_field_fn' which calculates the reward!
        
        def intercepted_reward_fn(x, t):
            reward = vector_field_fn(x, t)
            # --- DIAGNOSTIC LOG ---
            norm = torch.norm(reward.reshape(reward.shape[0], -1), dim=1).mean()
            if np.random.rand() < 0.05: # Print 5% of the time to avoid spam
                print(f"   [Diagnostic] Avg Entropy Force (Reward Gradient) Norm: {norm.item():.2f}")
            return reward
            
        return super().solve_vector_field(x_0, intercepted_reward_fn, prompt_emb)

# APPLY PATCH
smeme_module.VectorFieldAdjointSolver = DiagnosticSolver

# ============================================================================
# 3. MAIN RUN
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print("--- RUNNING DIAGNOSTIC S-MEME ---")
    
    # 1. Prepare Data & Model
    raw_data = generate_unbalanced_data()
    dataset_tensor = torch.from_numpy(raw_data).to(device)
    
    base_model = construct_diffusion_model(
        inputs=dataset_tensor.cpu(),
        normalizer_type='standard',
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    # 2. Pre-train (Briefly, for demo)
    print("Pre-training...")
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(dataset_tensor), batch_size=256, shuffle=True)
    
    base_model.train()
    for _ in range(1000): # 1000 steps is enough for 2D
        batch = next(iter(loader))[0]
        optimizer.zero_grad()
        loss = base_model(batch)
        loss.backward()
        optimizer.step()
        
    original_model = copy.deepcopy(base_model)

    # 3. Setup S-MEME with GENTLE Schedule
    # FIX: Use [1.0, 0.9] instead of [1.0, 0.1] to prevent explosion
    print("Configuring Gentle S-MEME...")
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20)
    
    smeme_config = SMEMEConfig(
        num_smeme_iterations=2,
        alpha_schedule=(1.0, 0.8), # Gentle Schedule!
        am_config=am_config
    )
    
    wrapped_model = DiffusionModelAdapter(base_model, am_config).to(device)
    solver = smeme_module.SMEMESolver(base_model=wrapped_model, config=smeme_config)
    # Reduced LR
    solver.optimizer = torch.optim.AdamW(solver.current_model.parameters(), lr=1e-5)

    # 4. Train
    def noise_generator():
        while True: yield torch.randn(256, 2).to(device)
    
    train_loader = iter(noise_generator())
    
    # We manually run the loop to print Gradient Norms
    print("\nStarting Fine-tuning (Watch the stats)...")
    
    # We call the solver.train but we limit it via the loader class trick
    class LimitedLoader:
        def __init__(self, gen, limit): self.gen = gen; self.limit = limit
        def __iter__(self): self.cnt=0; return self
        def __next__(self):
            if self.cnt >= self.limit: raise StopIteration
            self.cnt += 1
            return next(self.gen)
            
    finetuned_wrapper = solver.train(LimitedLoader(train_loader, 200)) # 200 steps per iter

    # 5. VISUALIZATION
    print("\nGenerating Diagnostic Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Vector Field of Base Model (The "Force")
    # This reveals if the gradient sign is correct! 
    # Arrows should point AWAY from the high density (5,5) and TOWARDS the void.
    plot_vector_field(wrapped_model, device, "Entropy Force Field (Base Model)", axes[0])
    
    # Plot 2: Base Samples
    samples_base = original_model.sample(batch_size=2000).cpu().numpy()
    sns.kdeplot(x=samples_base[:,0], y=samples_base[:,1], fill=True, cmap="Blues", ax=axes[1])
    axes[1].set_title("Base Model Density")
    axes[1].set_xlim(-10, 10); axes[1].set_ylim(-10, 10)
    
    # Plot 3: S-MEME Samples
    samples_smeme = finetuned_wrapper.model.sample(batch_size=2000).cpu().numpy()
    sns.kdeplot(x=samples_smeme[:,0], y=samples_smeme[:,1], fill=True, cmap="Oranges", ax=axes[2])
    axes[2].set_title("S-MEME Density (Gentle)")
    axes[2].set_xlim(-10, 10); axes[2].set_ylim(-10, 10)
    
    plt.savefig("smeme_diagnostics.png")
    print("Diagnostics saved to 'smeme_diagnostics.png'")
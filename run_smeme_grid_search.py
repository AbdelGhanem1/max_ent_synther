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

# ============================================================================
# 1. ROBUST DATA GENERATION (Far Apart 70/30)
# ============================================================================
def generate_unbalanced_data(n_samples=10000):
    # Mode A: 70% at (-4, -4)
    n_a = int(0.70 * n_samples)
    data_a = np.random.normal(loc=[-4.0, -4.0], scale=1.0, size=(n_a, 2))
    
    # Mode B: 30% at (4, 4)
    n_b = n_samples - n_a
    data_b = np.random.normal(loc=[4.0, 4.0], scale=1.0, size=(n_b, 2))
    
    return np.vstack([data_a, data_b]).astype(np.float32)

def check_balance(samples):
    # Simple distance check
    dist_a = np.linalg.norm(samples - np.array([-4.0, -4.0]), axis=1)
    dist_b = np.linalg.norm(samples - np.array([4.0, 4.0]), axis=1)
    
    count_a = np.sum(dist_a < 3.0) 
    count_b = np.sum(dist_b < 3.0)
    
    total = len(samples)
    return count_a / total, count_b / total

def plot_final_comparison(base_samples, smeme_samples):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    limits = ((-8, 8), (-8, 8))
    
    # Base
    ra, rb = check_balance(base_samples)
    sns.kdeplot(x=base_samples[:,0], y=base_samples[:,1], fill=True, cmap="Blues", ax=axes[0], levels=10)
    axes[0].set_title(f"Base Model (Imitator)\nA: {ra*100:.1f}% | B: {rb*100:.1f}%")
    axes[0].set_xlim(limits[0]); axes[0].set_ylim(limits[1])
    axes[0].scatter([-4, 4], [-4, 4], marker='x', color='red', s=100, label='True Centers')

    # S-MEME
    rsa, rsb = check_balance(smeme_samples)
    sns.kdeplot(x=smeme_samples[:,0], y=smeme_samples[:,1], fill=True, cmap="Oranges", ax=axes[1], levels=10)
    axes[1].set_title(f"S-MEME (Re-Balancer)\nA: {rsa*100:.1f}% | B: {rsb*100:.1f}%")
    axes[1].set_xlim(limits[0]); axes[1].set_ylim(limits[1])
    axes[1].scatter([-4, 4], [-4, 4], marker='x', color='red', s=100)

    plt.tight_layout()
    plt.savefig("smeme_robust_result.png")
    print("\nSaved plot to 'smeme_robust_result.png'")

# ============================================================================
# 2. MAIN LOGIC
# ============================================================================
if __name__ == "__main__":
    device = get_device()
    print(">>> Generating 70/30 Data...")
    dataset_np = generate_unbalanced_data()
    dataset = torch.from_numpy(dataset_np).to(device)
    
    # --- STEP 1: FORCE BASE MODEL TO LEARN ---
    print("\n>>> Constructing Base Model...")
    # We stick to the default network but we will train it PROPERLY
    base_model = construct_diffusion_model(
        inputs=dataset.cpu(), 
        normalizer_type='standard', 
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(dataset), batch_size=512, shuffle=True)
    
    print("\n>>> Training Base Model (Target: Capture Mode B > 20%)...")
    
    # We loop until we pass the quality check
    max_epochs = 100
    steps_per_epoch = 500
    
    for epoch in range(max_epochs):
        base_model.train()
        loss_accum = 0
        for _ in range(steps_per_epoch):
            # Random sampling from dataset is faster than iterator overhead for small data
            idx = torch.randint(0, len(dataset), (512,))
            batch = dataset[idx]
            
            opt.zero_grad()
            loss = base_model(batch)
            loss.backward()
            opt.step()
            loss_accum += loss.item()
            
        # Quality Check
        base_model.eval()
        with torch.no_grad():
            samples = base_model.sample(batch_size=2000).cpu().numpy()
        
        ra, rb = check_balance(samples)
        print(f"   [Epoch {epoch+1}] Loss: {loss_accum/steps_per_epoch:.4f} | Mode A: {ra*100:.1f}% | Mode B: {rb*100:.1f}%")
        
        if rb > 0.20: # If we capture at least 20% of Mode B (Target is 30%), we are good
            print(f">>> SUCCESS: Base model learned both modes (B={rb*100:.1f}%). Stopping pre-training.")
            break
            
    if rb < 0.20:
        print(">>> CRITICAL FAILURE: Base model failed to learn Mode B even after extensive training.")
        print(">>> Exiting to avoid wasting time on S-MEME.")
        exit()

    # --- STEP 2: RUN S-MEME ---
    print("\n>>> Running S-MEME (Paper Schedule)...")
    # Using the exact paper schedule that worked for exploration
    alphas = [10.0, 5.0, 2.0] # Multipliers: 0.1, 0.2, 0.5
    
    am_config = AdjointMatchingConfig(num_train_timesteps=1000, num_inference_steps=20)
    current_model = copy.deepcopy(base_model)
    prev_model = copy.deepcopy(current_model); prev_model.requires_grad_(False)
    
    opt_s = torch.optim.AdamW(current_model.parameters(), lr=1e-5)
    
    # Define Gradient Function
    def get_entropy_grad(x, t, model_snapshot):
        # We need a temporary wrapper to call _get_score_at_data
        # This is a bit ugly but ensures we use the exact logic from smeme_solver.py
        wrapper = DiffusionModelAdapter(model_snapshot, am_config).to(device)
        helper = smeme_module.SMEMESolver(wrapper, SMEMEConfig())
        
        # We also need to wrap the input 'x' logic if needed, but _get_score_at_data handles it
        # Actually, _get_score_at_data expects the 'model' arg to be the wrapper
        return helper._get_score_at_data(wrapper, x, t)

    for k, alpha in enumerate(alphas):
        print(f"   S-MEME Iteration {k+1} (Alpha {alpha})...")
        am_config.reward_multiplier = 1.0 / alpha
        
        # We implement the training loop directly here to avoid Wrapper hell
        # We must update the 'wrapped' models every iteration
        wrapped_curr = DiffusionModelAdapter(current_model, am_config).to(device)
        wrapped_prev = DiffusionModelAdapter(prev_model, am_config).to(device)
        
        solver = smeme_module.VectorFieldAdjointSolver(model_pre=wrapped_prev, model_fine=wrapped_curr, config=am_config)
        
        # Helper for gradient
        helper_dummy = smeme_module.SMEMESolver(wrapped_curr, SMEMEConfig())
        def entropy_grad(x, t): return helper_dummy._get_score_at_data(wrapped_prev, x, t)
        
        current_model.train()
        for _ in tqdm(range(500)):
            noise = torch.randn(512, 2).to(device)
            opt_s.zero_grad()
            loss = solver.solve_vector_field(noise, entropy_grad)
            loss.backward()
            opt_s.step()
            
        # Update Previous Model
        prev_model.load_state_dict(current_model.state_dict())
        
        # Quick check
        current_model.eval()
        with torch.no_grad():
            s_check = current_model.sample(batch_size=1000).cpu().numpy()
        ra, rb = check_balance(s_check)
        print(f"      -> Current Balance: A={ra*100:.1f}% | B={rb*100:.1f}%")

    # --- STEP 3: FINAL RESULTS ---
    print("\n>>> Generating Final Plot...")
    current_model.eval()
    with torch.no_grad():
        final_samples = current_model.sample(batch_size=5000).cpu().numpy()
        
    plot_final_comparison(samples, final_samples) # 'samples' is from Base, 'final_samples' from S-MEME
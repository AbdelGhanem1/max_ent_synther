import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import gymnasium as gym
from tqdm import tqdm
import copy
import sys
import gin

# Import your custom modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. DATA LOADING & SAMPLING
# ============================================================================

def load_models_and_data(args, device):
    """Loads models and real data."""
    print(f"Loading dataset {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    
    # We visualize Observations (States) primarily, as that shows coverage
    real_data = obs 
    
    # Input dim for the model (Obs + Act + ...)
    # Structure from training: [obs, act, rew, next_obs, term]
    input_dim = obs.shape[1] + act.shape[1] + 1 + obs.shape[1] + 1
    obs_dim = obs.shape[1]
    
    # Construct Models
    dummy_inputs = torch.randn(10, input_dim)
    base_model = construct_diffusion_model(inputs=dummy_inputs)
    smeme_model = construct_diffusion_model(inputs=dummy_inputs)
    
    # Load Weights
    print("Loading checkpoints...")
    base_ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_model.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)
    base_model.to(device)
    base_model.eval()
    
    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location='cpu', weights_only=False)
    smeme_model.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    smeme_model.to(device)
    smeme_model.eval()
    
    return base_model, smeme_model, real_data, obs_dim

def generate_samples(model, num_samples, batch_size=512):
    """
    Generates pure observation samples from the model.
    Dynamically fetches step count from Gin config to match training/eval.
    """
    model.eval()
    samples_list = []
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # --- DYNAMIC CONFIG CHECK ---
    # We attempt to find the correct step count from the loaded Gin config.
    # This prevents the "Vis=64 vs Eval=1000" mismatch.
    try:
        # 1. Try to get it from the solver config directly if possible
        if hasattr(model, 'solver') and hasattr(model.solver, 'num_inference_steps'):
             n_steps = model.solver.num_inference_steps
        # 2. Fallback to querying Gin explicitly
        else:
             n_steps = gin.query_parameter('AdjointMatchingConfig.num_inference_steps')
             
        # 3. Handle Gin returning macro strings or None
        if n_steps is None or isinstance(n_steps, str):
            print("⚠️ Warning: Could not resolve steps from Gin/Model. Defaulting to 60.")
            n_steps = 60
            
    except Exception as e:
        print(f"⚠️ Error reading config for steps ({e}). Defaulting to 1000.")
        n_steps = 60

    print(f"Generating {num_samples} samples using {n_steps} steps...")
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            # Pass the dynamic n_steps variable
            batch_samples = model.sample(batch_size=batch_size, num_sample_steps=n_steps)
            samples_list.append(batch_samples.cpu().numpy())
            
    # Crop and extract ONLY Observations (indices 0 to obs_dim)
    full_data = np.concatenate(samples_list, axis=0)[:num_samples]
    return full_data

# ============================================================================
# 2. UMAP VISUALIZATION
# ============================================================================

def run_visualization(args):
    device = get_device()
    
    # 1. Load Resources
    base_model, smeme_model, real_data_full, obs_dim = load_models_and_data(args, device)
    
    # 2. Prepare Data Batches
    N = args.num_points
    
    # A. Real Data (Subsampled)
    indices = np.random.choice(len(real_data_full), N, replace=False)
    data_real = real_data_full[indices]
    
    # B. Base Model Samples
    print("\n--- Sampling Base Model ---")
    raw_base = generate_samples(base_model, N)
    data_base = raw_base[:, :obs_dim] # Extract Obs only
    
    # C. S-MEME Samples
    print("\n--- Sampling S-MEME Model ---")
    raw_smeme = generate_samples(smeme_model, N)
    data_smeme = raw_smeme[:, :obs_dim] # Extract Obs only
    
    # 3. Combine for UMAP
    # We stack them to ensure they are projected into the SAME 2D space
    combined_data = np.vstack([data_real, data_base, data_smeme])
    
    print(f"\nFitting UMAP on {len(combined_data)} points...")
    reducer = umap.UMAP(
        n_neighbors=30,    # Higher = more global structure
        min_dist=0.3,      # Higher = looser clusters
        metric='euclidean',
        random_state=42
    )
    embedding = reducer.fit_transform(combined_data)
    
    # Split back
    emb_real = embedding[:N]
    emb_base = embedding[N:2*N]
    emb_smeme = embedding[2*N:]
    
    # 4. Plotting
    print("Generating Plot...")
    plt.figure(figsize=(12, 8), dpi=150)
    sns.set_style("whitegrid")
    
    # Plot Real Data (Background, Gray)
    plt.scatter(emb_real[:, 0], emb_real[:, 1], 
                c='lightgray', label='Original Dataset', 
                alpha=0.3, s=15, edgecolors='none')
    
    # Plot Base Model (Blue)
    plt.scatter(emb_base[:, 0], emb_base[:, 1], 
                c='royalblue', label='Base Model', 
                alpha=0.6, s=10, marker='o')
    
    # Plot S-MEME (Red/Orange - On Top)
    plt.scatter(emb_smeme[:, 0], emb_smeme[:, 1], 
                c='crimson', label='S-MEME (Finetuned)', 
                alpha=0.6, s=10, marker='x')
    
    plt.title(f"Manifold Visualization (UMAP): {args.dataset}", fontsize=16)
    plt.legend(markerscale=3.0)
    plt.tight_layout()
    
    save_path = "results_smeme/manifold_umap.png"
    plt.savefig(save_path)
    print(f"\n✅ Visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # HACK: Register globals for pickle
    try:
        sys.modules['__main__'].SMEMEConfig = SMEMEConfig
        sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    parser.add_argument('--num_points', type=int, default=2000, help="Points per category")
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    
    run_visualization(args)
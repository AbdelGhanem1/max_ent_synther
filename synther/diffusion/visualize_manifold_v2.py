# synther/diffusion/visualize_manifold_v2.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import gin
import sys
from matplotlib.lines import Line2D
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

# ============================================================================
# NEURIPS STYLE CONFIG
# ============================================================================
sns.set_context("paper", font_scale=1.6)
sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.15"})
plt.rcParams.update({
    "font.family": "serif", 
    "text.usetex": False,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5
})

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_obs_samples(model, num_samples, obs_dim, num_inference_steps):
    """Generates samples using the specified number of diffusion steps."""
    model.eval()
    samples = []
    bs = 1000  
    iters = int(np.ceil(num_samples/bs))
    
    print(f"  -> Sampling {num_samples} points ({num_inference_steps} steps)...")
    
    with torch.no_grad():
        for _ in range(iters):
            batch = model.sample(batch_size=bs, num_sample_steps=num_inference_steps)
            samples.append(batch.cpu().numpy()[:, :obs_dim])
            
    return np.vstack(samples)[:num_samples]

def visualize(args):
    device = get_device()
    
    # 1. LOAD DATA & MODELS
    print(f"Loading Dataset: {args.dataset}...")
    dataset = d4rl_offline_dataset(args.dataset)
    real_obs = dataset['observations']
    
    input_dim = real_obs.shape[1] + dataset['actions'].shape[1] + 1 + real_obs.shape[1] + 1
    
    dummy = torch.randn(2, input_dim)
    base = construct_diffusion_model(inputs=dummy).to(device)
    smeme = construct_diffusion_model(inputs=dummy).to(device)
    
    print("Loading Checkpoints...")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)

    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location=device, weights_only=False)
    smeme.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    
    # 2. PREPARE DATA
    # Use a MASSIVE amount of real data to define the manifold support
    N_real = min(len(real_obs), args.num_real_points)
    N_gen = args.num_generated_points
    
    print(f"Preparing Data for UMAP:")
    print(f"  - Real Data (Background): {N_real} points")
    print(f"  - Generated Models:       {N_gen} points each")
    
    idx = np.random.choice(len(real_obs), N_real, replace=False)
    data_real = real_obs[idx]
    
    data_base = generate_obs_samples(base, N_gen, real_obs.shape[1], args.num_inference_steps)
    data_smeme = generate_obs_samples(smeme, N_gen, real_obs.shape[1], args.num_inference_steps)
    
    # 3. UMAP PROJECTION
    print("Computing UMAP Projection (Training on combined density)...")
    # We train UMAP on everything so the spaces are aligned
    all_data = np.vstack([data_real, data_base, data_smeme])
    
    # n_neighbors=50 preserves global structure better (good for 'full support' view)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=42)
    embedding = reducer.fit_transform(all_data)
    
    # Slice back out
    emb_real = embedding[:N_real]
    emb_base = embedding[N_real : N_real+N_gen]
    emb_smeme = embedding[N_real+N_gen :]

    # COLORS
    colors = sns.color_palette("colorblind", 10) 
    c_real = colors[7]  # Grey
    c_base = colors[0]  # Blue
    c_smeme = colors[3] # Orange/Reddish

    # ========================================================================
    # FIGURE 1: SCATTER PLOT
    # ========================================================================
    print("Generating Figure 1: Scatter Plot...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # A. Real Data: High count, very low alpha -> Creates a "texture" or "blur"
    ax1.scatter(emb_real[:,0], emb_real[:,1], c=[c_real], s=5, alpha=0.05, label='Offline Data (Full Support)', rasterized=True)
    
    # B. Models: Larger dots, higher alpha
    ax1.scatter(emb_base[:,0], emb_base[:,1], c=[c_base], s=20, alpha=0.6, marker='o', label='Base Diffusion', rasterized=True)
    ax1.scatter(emb_smeme[:,0], emb_smeme[:,1], c=[c_smeme], s=30, alpha=0.8, marker='x', label='S-MEME (Ours)', rasterized=True)
    
    ax1.set_title(f"Manifold Coverage: {args.dataset}", fontweight='bold', pad=15)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    legend_elements_1 = [
        Line2D([0], [0], marker='o', color='w', label='Offline Data', markerfacecolor=c_real, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Base Model', markerfacecolor=c_base, markersize=10),
        Line2D([0], [0], marker='x', color=c_smeme, label='S-MEME (Ours)', markersize=10, lw=2)
    ]
    ax1.legend(handles=legend_elements_1, loc='upper right', frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig("manifold_scatter.pdf", dpi=300, bbox_inches='tight')
    print("Saved manifold_scatter.pdf")
    plt.close(fig1)

    # ========================================================================
    # FIGURE 2: DENSITY PLOT
    # ========================================================================
    print("Generating Figure 2: Density Plot...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # A. Real Data: Filled Contour (The "Land")
    # levels=10 gives a very detailed map of the support
    sns.kdeplot(x=emb_real[:,0], y=emb_real[:,1], levels=10, color=c_real, fill=True, alpha=0.2, ax=ax2, thresh=0.02)
    
    # B. Models: Lines only (The "Explorers")
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], levels=4, color=c_base, linewidths=2.5, ax=ax2, thresh=0.1)
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], levels=4, color=c_smeme, linewidths=3.0, linestyles='--', ax=ax2, thresh=0.1)
    
    ax2.set_title(f"Manifold Density: {args.dataset}", fontweight='bold', pad=15)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    legend_elements_2 = [
        Line2D([0], [0], color=c_real, lw=4, alpha=0.5, label='Offline Data Support'),
        Line2D([0], [0], color=c_base, lw=2.5, label='Base Model'),
        Line2D([0], [0], color=c_smeme, lw=3.0, linestyle='--', label='S-MEME (Ours)')
    ]
    ax2.legend(handles=legend_elements_2, loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("manifold_density.pdf", dpi=300, bbox_inches='tight')
    print("Saved manifold_density.pdf")
    plt.close(fig2)

if __name__ == "__main__":
    sys.modules['__main__'].SMEMEConfig = SMEMEConfig
    sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    
    # Tweaked Defaults
    parser.add_argument('--num_inference_steps', type=int, default=64)
    parser.add_argument('--num_real_points', type=int, default=50000, help="High count for background")
    parser.add_argument('--num_generated_points', type=int, default=3000, help="Moderate count for generation")
    
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    visualize(args)
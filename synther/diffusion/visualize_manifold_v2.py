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

# Config for NeurIPS quality
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {"grid.linestyle": "--", "axes.edgecolor": "0.15"})
plt.rcParams.update({
    "font.family": "serif",  # Use serif/latex-like fonts
    "text.usetex": False,    # Set True if you have full TeX installed
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_obs_samples(model, num_samples, obs_dim):
    model.eval()
    samples = []
    # Sample in batches
    bs = 500
    iters = int(np.ceil(num_samples/bs))
    with torch.no_grad():
        for _ in range(iters):
            batch = model.sample(batch_size=bs, num_sample_steps=64)
            samples.append(batch.cpu().numpy()[:, :obs_dim]) # Keep only observations
    return np.vstack(samples)[:num_samples]

def visualize(args):
    device = get_device()
    
    # 1. Load Data & Models (Simplified loading logic)
    dataset = d4rl_offline_dataset(args.dataset)
    real_obs = dataset['observations']
    input_dim = real_obs.shape[1] + dataset['actions'].shape[1] + 1 + real_obs.shape[1] + 1
    
    dummy = torch.randn(2, input_dim)
    base = construct_diffusion_model(inputs=dummy).to(device)
    smeme = construct_diffusion_model(inputs=dummy).to(device)
    
    # Load
    base.load_state_dict(torch.load(args.base_checkpoint, map_location=device)['model'])
    smeme.load_state_dict(torch.load(args.smeme_checkpoint, map_location=device)['model'])
    
    # 2. Generate
    print("Generating Samples for Visualization...")
    N = 2000
    idx = np.random.choice(len(real_obs), N, replace=False)
    data_real = real_obs[idx]
    data_base = generate_obs_samples(base, N, real_obs.shape[1])
    data_smeme = generate_obs_samples(smeme, N, real_obs.shape[1])
    
    # 3. UMAP Projection
    print("Projecting with UMAP...")
    all_data = np.vstack([data_real, data_base, data_smeme])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(all_data)
    
    emb_real = embedding[:N]
    emb_base = embedding[N:2*N]
    emb_smeme = embedding[2*N:]
    
    # 4. Professional Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Colors (Colorblind friendly)
    colors = sns.color_palette("colorblind", 3)
    c_real, c_base, c_smeme = colors[7], colors[0], colors[3] # Grey, Blue, Orange-ish
    
    # A. Real Data: Density Contour (Background) + Scatter
    sns.kdeplot(x=emb_real[:,0], y=emb_real[:,1], 
                levels=5, color=c_real, fill=True, alpha=0.15, ax=ax)
    ax.scatter(emb_real[:,0], emb_real[:,1], c=[c_real], s=10, alpha=0.2, label='Offline Dataset', rasterized=True)
    
    # B. Base Model: Scatter Only (or Contour)
    # ax.scatter(emb_base[:,0], emb_base[:,1], c=[c_base], s=15, marker='o', alpha=0.5, label='Base Diffusion')
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], levels=3, color=c_base, linewidths=1.5, ax=ax)
    
    # C. S-MEME: Scatter + Contour (Highlight expansion)
    ax.scatter(emb_smeme[:,0], emb_smeme[:,1], c=[c_smeme], s=20, marker='x', alpha=0.8, label='S-MEME (Ours)')
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], levels=3, color=c_smeme, linewidths=2.0, linestyles='--', ax=ax)
    
    # Styling
    ax.set_title(f"Manifold Exploration: {args.dataset}", fontweight='bold', pad=15)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # Custom Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Offline Data', markerfacecolor=c_real, markersize=8),
        Line2D([0], [0], color=c_base, lw=2, label='Base Diffusion (Density)'),
        Line2D([0], [0], marker='x', color=c_smeme, label='S-MEME (Ours)', markersize=8, lw=0)
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9)
    
    # Remove top/right spines
    sns.despine()
    
    # Save
    plt.tight_layout()
    plt.savefig("manifold_neurips.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("manifold_neurips.png", dpi=300, bbox_inches='tight')
    print("Saved to manifold_neurips.pdf")

if __name__ == "__main__":
    sys.modules['__main__'].SMEMEConfig = SMEMEConfig
    sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    visualize(args)
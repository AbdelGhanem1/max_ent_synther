# synther/diffusion/visualize_manifold_final.py

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
# PAPER-QUALITY CONFIGURATION
# ============================================================================
sns.set_context("paper", font_scale=1.8)
sns.set_style("white", {"axes.edgecolor": "0.15", "grid.color": "0.9"})
plt.rcParams.update({
    "font.family": "serif", 
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 3.0
})

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_obs_samples(model, num_samples, obs_dim, num_inference_steps):
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
    
    # 2. GENERATE SAMPLES
    N_real = min(len(real_obs), args.num_real_points)
    N_gen = args.num_generated_points
    
    idx = np.random.choice(len(real_obs), N_real, replace=False)
    data_real = real_obs[idx]
    
    data_base = generate_obs_samples(base, N_gen, real_obs.shape[1], args.num_inference_steps)
    data_smeme = generate_obs_samples(smeme, N_gen, real_obs.shape[1], args.num_inference_steps)
    
    # 3. UMAP
    print("Computing UMAP...")
    all_data = np.vstack([data_real, data_base, data_smeme])
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=42)
    embedding = reducer.fit_transform(all_data)
    
    # FIX: Correct slicing using N_real and N_gen
    emb_real = embedding[:N_real]
    emb_base = embedding[N_real : N_real+N_gen]
    emb_smeme = embedding[N_real+N_gen :]

    # COLORS
    colors = sns.color_palette("colorblind", 10) 
    c_real = colors[7]  # Grey
    c_base = colors[0]  # Blue
    c_smeme = colors[3] # Orange/Vermillion

    # ========================================================================
    # FIGURE 1: THE CLEAN COMPOSITE (Best for Paper)
    # ========================================================================
    print("Generating 'manifold_composite_lines.pdf'...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # 1. Real Data: Dashed Grey Boundary
    sns.kdeplot(x=emb_real[:,0], y=emb_real[:,1], thresh=0.05, levels=1, color=c_real, 
                linewidths=2.0, linestyles="--", ax=ax1, fill=False)

    # 2. Base Model: Solid Blue Boundary
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], thresh=0.05, levels=1, color=c_base, 
                linewidths=3.0, ax=ax1, fill=False)
    
    # 3. S-MEME: Solid Orange Boundary
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], thresh=0.05, levels=1, color=c_smeme, 
                linewidths=3.0, ax=ax1, fill=False)

    ax1.set_title(f"Manifold Boundaries (95% Confidence)", fontweight='bold', pad=20)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    legend_1 = [
        Line2D([0], [0], color=c_real, lw=2.0, linestyle="--", label='Real Data Support'),
        Line2D([0], [0], color=c_base, lw=3.0, label='Base Model Support'),
        Line2D([0], [0], color=c_smeme, lw=3.0, label='S-MEME Support'),
    ]
    ax1.legend(handles=legend_1, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("manifold_composite_lines.pdf", dpi=300)
    plt.close(fig1)

    # ========================================================================
    # FIGURE 2: THE CORES (High Density Peaks Only)
    # ========================================================================
    print("Generating 'manifold_cores.pdf'...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Real Data (Background Reference)
    sns.kdeplot(x=emb_real[:,0], y=emb_real[:,1], thresh=0.05, levels=1, color=c_real, 
                fill=True, alpha=0.1, ax=ax2)
    
    # Base Model Core (Blue) - Peaks only (Top 50%)
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], thresh=0.5, levels=1, color=c_base, 
                fill=True, alpha=0.5, ax=ax2)
    
    # S-MEME Core (Orange) - Peaks only (Top 50%)
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], thresh=0.5, levels=1, color=c_smeme, 
                fill=True, alpha=0.5, ax=ax2)
    
    ax2.set_title(f"High-Density Modes (Top 50%)", fontweight='bold', pad=20)
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    legend_2 = [
        Line2D([0], [0], color=c_real, lw=10, alpha=0.2, label='Real Data (Global)'),
        Line2D([0], [0], color=c_base, lw=10, alpha=0.5, label='Base Model Core'),
        Line2D([0], [0], color=c_smeme, lw=10, alpha=0.5, label='S-MEME Core'),
    ]
    ax2.legend(handles=legend_2, loc='upper right')
    plt.tight_layout()
    plt.savefig("manifold_cores.pdf", dpi=300)
    plt.close(fig2)

    # ========================================================================
    # FIGURE 3: THE SCATTER (Raw Reference)
    # ========================================================================
    print("Generating 'manifold_scatter.pdf'...")
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    ax3.scatter(emb_real[:,0], emb_real[:,1], c=[c_real], s=2, alpha=0.05)
    ax3.scatter(emb_base[:,0], emb_base[:,1], c=[c_base], s=15, alpha=0.4, label='Base')
    ax3.scatter(emb_smeme[:,0], emb_smeme[:,1], c=[c_smeme], s=25, alpha=0.5, marker='x', label='S-MEME')
    ax3.legend(loc='upper right')
    ax3.set_xticks([])
    ax3.set_yticks([])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("manifold_scatter.pdf", dpi=300)
    plt.close(fig3)
    
    print("\n✅ DONE. Generated 3 plots.")

if __name__ == "__main__":
    sys.modules['__main__'].SMEMEConfig = SMEMEConfig
    sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    
    parser.add_argument('--num_inference_steps', type=int, default=64)
    parser.add_argument('--num_real_points', type=int, default=50000)
    parser.add_argument('--num_generated_points', type=int, default=3000)
    
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    visualize(args)
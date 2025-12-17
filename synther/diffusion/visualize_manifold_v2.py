# synther/diffusion/visualize_manifold_v3.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import gin
import sys
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

# Clean, professional styling
sns.set_context("paper", font_scale=1.6)
sns.set_style("white", {"axes.edgecolor": "0.15"}) # White background is cleaner for overlaps
plt.rcParams.update({
    "font.family": "serif", 
    "text.usetex": False,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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
    
    # 1. LOAD DATA
    print(f"Loading Dataset: {args.dataset}...")
    dataset = d4rl_offline_dataset(args.dataset)
    real_obs = dataset['observations']
    input_dim = real_obs.shape[1] + dataset['actions'].shape[1] + 1 + real_obs.shape[1] + 1
    
    # 2. LOAD MODELS
    dummy = torch.randn(2, input_dim)
    base = construct_diffusion_model(inputs=dummy).to(device)
    smeme = construct_diffusion_model(inputs=dummy).to(device)
    
    print("Loading Checkpoints...")
    base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)

    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location=device, weights_only=False)
    smeme.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    
    # 3. PREPARE DATA
    N_real = min(len(real_obs), args.num_real_points)
    N_gen = args.num_generated_points
    
    idx = np.random.choice(len(real_obs), N_real, replace=False)
    data_real = real_obs[idx]
    
    data_base = generate_obs_samples(base, N_gen, real_obs.shape[1], args.num_inference_steps)
    data_smeme = generate_obs_samples(smeme, N_gen, real_obs.shape[1], args.num_inference_steps)
    
    # 4. UMAP
    print("Computing UMAP Projection...")
    all_data = np.vstack([data_real, data_base, data_smeme])
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=42)
    embedding = reducer.fit_transform(all_data)
    
    emb_real = embedding[:N_real]
    emb_base = embedding[N_real : N_real+N_gen]
    emb_smeme = embedding[N_real+N_gen :]

    # COLORS
    colors = sns.color_palette("colorblind", 10) 
    c_real = colors[7]  # Grey
    c_base = colors[0]  # Blue
    c_smeme = colors[3] # Orange
    
    # ========================================================================
    # GENERATE DENSITY PLOT (FILLED OVERLAP STYLE)
    # ========================================================================
    print("Generating 'manifold_density_filled.pdf'...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # A. Real Data: THIN OUTLINE ONLY (The Boundary)
    # We do NOT fill this, so it doesn't muddy the colors. It acts as a fence.
    sns.kdeplot(x=emb_real[:,0], y=emb_real[:,1], levels=1, color=c_real, 
                linewidths=1.5, ax=ax, thresh=0.05, linestyle="--")
    
    # B. Base Model: FILLED BLUE
    # alpha=0.4 allows the background to show through slightly
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], levels=5, color=c_base, 
                fill=True, alpha=0.4, ax=ax, thresh=0.1)
    
    # C. S-MEME: FILLED ORANGE
    # alpha=0.4 on top of Blue creates a specific "Overlap Color"
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], levels=5, color=c_smeme, 
                fill=True, alpha=0.4, ax=ax, thresh=0.1)
    
    ax.set_title(f"Manifold Expansion: {args.dataset}", fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    # Custom Legend interpreting the overlaps
    legend_elements = [
        Line2D([0], [0], color=c_real, lw=1.5, linestyle="--", label='Real Data Boundary'),
        Patch(facecolor=c_base, alpha=0.4, label='Base Model (Conservative)'),
        Patch(facecolor=c_smeme, alpha=0.4, label='S-MEME (Expansion)'),
        # Manually creating a "Patch" that represents the overlap color (approx visual)
        Patch(facecolor='purple', alpha=0.4, label='Overlap (Retained Knowledge)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig("manifold_density_filled.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("manifold_density_filled.png", dpi=300, bbox_inches='tight')
    print("✅ Saved to manifold_density_filled.pdf")

    # ========================================================================
    # GENERATE SCATTER PLOT (Optional, for reference)
    # ========================================================================
    print("Generating 'manifold_scatter.pdf'...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.scatter(emb_real[:,0], emb_real[:,1], c=[c_real], s=2, alpha=0.05, label='Offline Data')
    ax2.scatter(emb_base[:,0], emb_base[:,1], c=[c_base], s=15, alpha=0.5, label='Base Model')
    ax2.scatter(emb_smeme[:,0], emb_smeme[:,1], c=[c_smeme], s=25, alpha=0.6, marker='x', label='S-MEME')
    ax2.legend()
    ax2.set_xticks([])
    ax2.set_yticks([])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig("manifold_scatter.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig2)

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
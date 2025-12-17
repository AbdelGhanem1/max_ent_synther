# synther/diffusion/visualize_manifold_stable.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import gin
import sys
import random
import os
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

def set_seed(seed):
    print(f"Setting global seed to {seed}...")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    set_seed(args.seed)
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
    # Use MORE real points to build a stable map
    N_real = min(len(real_obs), 50000) 
    N_gen = args.num_generated_points
    
    idx = np.random.choice(len(real_obs), N_real, replace=False)
    data_real = real_obs[idx]
    
    data_base = generate_obs_samples(base, N_gen, real_obs.shape[1], args.num_inference_steps)
    data_smeme = generate_obs_samples(smeme, N_gen, real_obs.shape[1], args.num_inference_steps)
    
    # 3. UMAP (THE STABLE METHOD)
    print("Computing Stable UMAP...")
    print("  1. Fitting Map on Real Data ONLY (The Anchor)...")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.5, random_state=args.seed)
    reducer.fit(data_real) # <--- CRITICAL CHANGE: Only fit on real data
    
    print("  2. Projecting Real Data...")
    emb_real = reducer.transform(data_real)
    
    print("  3. Projecting Base Model...")
    emb_base = reducer.transform(data_base)
    
    print("  4. Projecting S-MEME...")
    emb_smeme = reducer.transform(data_smeme)

    # Note: We downsample Real Data for plotting so the PDF doesn't explode
    plot_idx = np.random.choice(len(emb_real), min(len(emb_real), 10000), replace=False)
    emb_real_plot = emb_real[plot_idx]

    # COLORS
    colors = sns.color_palette("colorblind", 10) 
    c_real = colors[7]  # Grey
    c_base = colors[0]  # Blue
    c_smeme = colors[3] # Orange/Vermillion

    # ========================================================================
    # FIGURE: COMPOSITE
    # ========================================================================
    print("Generating 'manifold_stable.pdf'...")
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    
    # Real Data
    sns.kdeplot(x=emb_real_plot[:,0], y=emb_real_plot[:,1], levels=[0.05], color=c_real, 
                linewidths=2.0, linestyles="--", ax=ax1, fill=False)

    # Base Model
    sns.kdeplot(x=emb_base[:,0], y=emb_base[:,1], levels=[0.05], color=c_base, 
                linewidths=3.0, ax=ax1, fill=False)
    
    # S-MEME
    sns.kdeplot(x=emb_smeme[:,0], y=emb_smeme[:,1], levels=[0.05], color=c_smeme, 
                linewidths=3.0, ax=ax1, fill=False)

    ax1.set_title(f"Stable Manifold Projection", fontweight='bold', pad=20)
    ax1.set_xticks([])
    ax1.set_yticks([])
    sns.despine(left=True, bottom=True)
    
    legend_1 = [
        Line2D([0], [0], color=c_real, lw=2.0, linestyle="--", label='Real Data'),
        Line2D([0], [0], color=c_base, lw=3.0, label='Base Model'),
        Line2D([0], [0], color=c_smeme, lw=3.0, label='S-MEME'),
    ]
    ax1.legend(handles=legend_1, loc='upper right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig("manifold_stable.pdf", dpi=300)
    print("✅ Saved to manifold_stable.pdf")

if __name__ == "__main__":
    sys.modules['__main__'].SMEMEConfig = SMEMEConfig
    sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    parser.add_argument('--num_inference_steps', type=int, default=64)
    parser.add_argument('--num_generated_points', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    visualize(args)
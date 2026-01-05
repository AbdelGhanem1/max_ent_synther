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

def run_unclipped_test():
    device = get_device()
    dataset = torch.from_numpy(generate_unbalanced_data()).to(device)
    
    print(">>> Training Base Model...")
    base_model = construct_diffusion_model(
        inputs=dataset.cpu(), 
        normalizer_type='standard', 
        denoising_network=ResidualMLPDenoiser
    ).to(device)
    
    # ... (Train base model for ~1000 steps) ...
    opt = torch.optim.Adam(base_model.parameters(), lr=1e-3)
    for _ in range(1000):
        indices = torch.randperm(len(dataset))[:256]
        opt.zero_grad(); base_model(dataset[indices]).backward(); opt.step()

    print("\n>>> Testing Sampler Behavior...")
    base_model.eval()
    
    # TEST 1: Standard Sampling
    print("1. Standard Sampling:")
    s1 = base_model.sample(batch_size=1000).cpu().numpy()
    print(f"   Min: {s1.min():.2f} | Max: {s1.max():.2f}")
    
    # TEST 2: Unclipped Sampling (Try passing the flag)
    print("2. Unclipped Sampling (clip_denoised=False):")
    try:
        # Most implementations accept kwargs for the sampler
        s2 = base_model.sample(batch_size=1000, clip_denoised=False).cpu().numpy()
        print(f"   Min: {s2.min():.2f} | Max: {s2.max():.2f}")
        
        if s2.max() > s1.max():
            print(">>> SUCCESS: Clipping disabled! S-MEME can now explore.")
        else:
            print(">>> FAIL: Flag ignored or bounds are natural.")
            
    except TypeError:
        print(">>> ERROR: 'clip_denoised' argument not accepted.")

if __name__ == "__main__":
    run_unclipped_test()
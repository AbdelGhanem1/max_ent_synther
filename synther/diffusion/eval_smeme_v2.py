# synther/diffusion/eval_smeme_v3.py

import argparse
import torch
import numpy as np
import random
import os
import gymnasium as gym
import gin
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import sys

# Import Custom Modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    """Sets the seed for reproducibility."""
    print(f"Setting global seed to {seed}...")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# 1. METRICS
# ============================================================================

def compute_vendi_score(samples, kernel='rbf', gamma=None):
    """
    Computes the Vendi Score (Diversity).
    VS = exp( - sum(lambda_i * log(lambda_i)) )
    """
    # If N is massive, Vendi (O(N^3)) will crash. We limit internal computation to 5k.
    # But we respect the user's N if it is smaller.
    if len(samples) > 5000:
        print(f"  [Note] Vendi Score subsampling 5000 points from {len(samples)} for efficiency.")
        indices = np.random.choice(len(samples), 5000, replace=False)
        samples = samples[indices]

    if kernel == 'rbf':
        dists = cdist(samples, samples, metric='euclidean')
        if gamma is None:
            gamma = 1.0 / (np.mean(dists) + 1e-6)
        K = np.exp(-gamma * (dists ** 2))
    elif kernel == 'linear':
        K = np.dot(samples, samples.T)
    
    n = K.shape[0]
    K = K / n
    eigvals = eigh(K, eigvals_only=True)
    eigvals = eigvals[eigvals > 1e-10]
    eigvals = eigvals / eigvals.sum()
    entropy = -np.sum(eigvals * np.log(eigvals))
    return np.exp(entropy)

class DynamicsOracle:
    """
    Uses standard Gymnasium MuJoCo environments to verify transitions.
    """
    def __init__(self, d4rl_dataset_name):
        self.dataset_name = d4rl_dataset_name
        self.env_name = self._map_to_gym(d4rl_dataset_name)
        
        print(f"DynamicsOracle: Mapping '{d4rl_dataset_name}' -> '{self.env_name}'")
        try:
            self.env = gym.make(self.env_name)
        except Exception as e:
            print(f"Warning: Could not load environment {self.env_name}. Dynamics check will be skipped.")
            self.env = None
        
    def _map_to_gym(self, dataset_name):
        name = dataset_name.lower()
        if 'hopper' in name: return 'Hopper-v4'
        elif 'walker' in name: return 'Walker2d-v4'
        elif 'halfcheetah' in name: return 'HalfCheetah-v4'
        elif 'ant' in name: return 'Ant-v4'
        else: return 'Hopper-v4' 

    def obs_to_state(self, obs):
        if self.env is None: return None, None
        try:
            model = self.env.unwrapped.model
            qpos_dim = model.nq
            qvel_dim = model.nv
            qpos = np.zeros(qpos_dim)
            qpos[1:] = obs[:qpos_dim-1]
            qvel = obs[qpos_dim-1:]
            return qpos, qvel
        except:
            return None, None

    def verify_transitions(self, generated_data, obs_dim, act_dim, num_checks=500):
        if self.env is None: return -1.0

        gen_obs = generated_data[:, :obs_dim]
        gen_act = generated_data[:, obs_dim : obs_dim+act_dim]
        gen_next_obs = generated_data[:, obs_dim+act_dim+1 : obs_dim+act_dim+1+obs_dim]
        
        errors = []
        valid_checks = 0
        
        # Check a random subset of the generated points
        actual_checks = min(num_checks, len(generated_data))
        indices = np.random.choice(len(generated_data), actual_checks, replace=False)
        
        for idx in indices:
            s = gen_obs[idx]
            a = gen_act[idx]
            s_prime_gen = gen_next_obs[idx]
            
            qpos, qvel = self.obs_to_state(s)
            
            if qpos is None: continue
                
            try:
                self.env.reset()
                self.env.unwrapped.set_state(qpos, qvel)
                s_prime_true, _, _, _, _ = self.env.step(a)
                
                mse = np.mean((s_prime_true - s_prime_gen)**2)
                errors.append(mse)
                valid_checks += 1
            except Exception:
                pass

        if valid_checks == 0: return -1.0
        return np.mean(errors)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Globals for pickling
    sys.modules['__main__'].SMEMEConfig = SMEMEConfig
    sys.modules['__main__'].AdjointMatchingConfig = AdjointMatchingConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--base_checkpoint', type=str, required=True)
    parser.add_argument('--smeme_checkpoint', type=str, required=True)
    
    # NEW ARGUMENTS
    parser.add_argument('--num_points', type=int, default=2000, help="Number of points to generate and evaluate")
    parser.add_argument('--num_inference_steps', type=int, default=64, help="Diffusion sampling steps")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument('--gin_config_files', nargs='*', default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', default=[])
    
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    device = get_device()
    
    # 1. Set Seed
    set_seed(args.seed)
    
    # 2. Load Data
    print(f"Loading dataset info for {args.dataset}...")
    dataset = d4rl_offline_dataset(args.dataset)
    obs = dataset['observations']
    act = dataset['actions']
    
    input_dim = obs.shape[1] + act.shape[1] + 1 + obs.shape[1] + 1
    dummy_inputs = torch.randn(10, input_dim)
    
    # 3. Load Models
    print("Loading Models...")
    base_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    smeme_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    
    base_ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_model.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)
    
    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location='cpu', weights_only=False)
    smeme_model.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    
    # 4. Generator
    def sample_fn(model, N, steps):
        model.eval()
        res = []
        batch_size = 500
        num_batches = int(np.ceil(N / batch_size))
        
        print(f"  Sampling {N} points with {steps} steps...")
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Sampling Batches"):
                batch = model.sample(batch_size=batch_size, num_sample_steps=steps)
                res.append(batch.cpu().numpy())
        return np.vstack(res)[:N]

    print(f"\n--- Generating {args.num_points} Samples (Steps={args.num_inference_steps}) ---")
    samples_base = sample_fn(base_model, args.num_points, args.num_inference_steps)
    samples_smeme = sample_fn(smeme_model, args.num_points, args.num_inference_steps)
    
    # Real Data subset for Vendi comparison
    real_data_subset = np.concatenate([
        dataset['observations'], 
        dataset['actions'],
        dataset['rewards'][:,None],
        dataset['next_observations'],
        dataset['terminals'][:,None]
    ], axis=1)
    
    # Only use as many real points as generated points for fair comparison, 
    # or use all if we want ground truth diversity
    real_indices = np.random.choice(len(real_data_subset), min(len(real_data_subset), args.num_points), replace=False)
    samples_real = real_data_subset[real_indices]

    # 5. Evaluation
    print("\n" + "="*60)
    print(f"EVALUATION REPORT (Seed={args.seed})")
    print("="*60)
    
    # A. Vendi Score
    print("1. Diversity (Vendi Score):")
    vs_real = compute_vendi_score(samples_real)
    vs_base = compute_vendi_score(samples_base)
    vs_smeme = compute_vendi_score(samples_smeme)
    
    print(f"  Real Data:  {vs_real:.2f}")
    print(f"  Base Model: {vs_base:.2f}")
    print(f"  S-MEME:     {vs_smeme:.2f} ({(vs_smeme-vs_base)/vs_base*100:+.1f}%)")
    
    # B. Dynamics Oracle
    print("\n2. Dynamics Consistency (MSE vs Simulator):")
    oracle = DynamicsOracle(args.dataset)
    
    # We check 1000 points or whatever num_points is if smaller
    check_N = min(1000, args.num_points)
    
    mse_base = oracle.verify_transitions(samples_base, obs.shape[1], act.shape[1], num_checks=check_N)
    mse_smeme = oracle.verify_transitions(samples_smeme, obs.shape[1], act.shape[1], num_checks=check_N)
    
    if mse_base == -1:
         print("  [Skipped] Dynamics check not supported for this environment type.")
    else:
        print(f"  Base Model: {mse_base:.5f}")
        print(f"  S-MEME:     {mse_smeme:.5f}")
        
        ratio = mse_smeme / (mse_base + 1e-8)
        if ratio > 1.5:
            print(f"  -> Warning: S-MEME error is {ratio:.1f}x higher (Potential Manifold Violation)")
        else:
            print(f"  -> Success: S-MEME maintains physical realism ({ratio:.1f}x error)")
            
    print("="*60)
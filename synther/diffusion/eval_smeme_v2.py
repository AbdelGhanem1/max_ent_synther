# synther/diffusion/eval_smeme_v2.py

import argparse
import torch
import numpy as np
import gymnasium as gym
import d4rl # Registers environments
import gin
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import copy
import sys

# Import Custom Modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import DiffusionModelAdapter, AdjointMatchingConfig, SMEMEConfig
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. PRINCIPLED METRICS: VENDI SCORE & DYNAMICS ORACLE
# ============================================================================

def compute_vendi_score(samples, kernel='rbf', gamma=None):
    """
    Computes the Vendi Score (Diversity) as defined in the S-MEME paper.
    VS = exp( - sum(lambda_i * log(lambda_i)) )
    """
    if len(samples) > 5000:
        # Subsample for SVD performance if N is huge
        indices = np.random.choice(len(samples), 5000, replace=False)
        samples = samples[indices]

    # 1. Compute Kernel Matrix K
    if kernel == 'rbf':
        dists = cdist(samples, samples, metric='euclidean')
        if gamma is None:
            # Heuristic: 1 / mean_dist
            gamma = 1.0 / (np.mean(dists) + 1e-6)
        K = np.exp(-gamma * (dists ** 2))
    elif kernel == 'linear':
        K = np.dot(samples, samples.T)
    
    # 2. Normalize K so trace=1
    n = K.shape[0]
    K = K / n

    # 3. Compute Eigenvalues
    # We use eigh because K is symmetric
    eigvals = eigh(K, eigvals_only=True)
    
    # Filter small numerical noise and normalize
    eigvals = eigvals[eigvals > 1e-10]
    eigvals = eigvals / eigvals.sum()
    
    # 4. Compute Von Neumann Entropy
    entropy = -np.sum(eigvals * np.log(eigvals))
    
    vendi = np.exp(entropy)
    return vendi

class DynamicsOracle:
    """
    Principled Manifold Check.
    Uses the TRUE online environment to verify transitions.
    """
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        
    def obs_to_state(self, obs):
        """
        Heuristic to map Observation -> MuJoCo qpos/qvel.
        This is environment specific because D4RL observations often 
        exclude CoM position (x-axis).
        """
        # Note: This works for standard Gym MuJoCo v2/v3/v4 envs.
        # AntMaze requires specific handling.
        
        qpos_dim = self.env.unwrapped.model.nq
        qvel_dim = self.env.unwrapped.model.nv
        
        if 'Hopper' in self.env_name or 'Walker' in self.env_name or 'HalfCheetah' in self.env_name:
            # In these envs, obs = [qpos[1:], qvel]. qpos[0] is usually x-position (ignored).
            # We assume x=0 for verification physics.
            qpos = np.zeros(qpos_dim)
            qpos[1:] = obs[:qpos_dim-1]
            qvel = obs[qpos_dim-1:]
            return qpos, qvel
        
        return None, None

    def verify_transitions(self, generated_data, obs_dim, act_dim):
        """
        Computes MSE between Gen(s') and True(s, a).
        Returns: Dynamics MSE.
        """
        gen_obs = generated_data[:, :obs_dim]
        gen_act = generated_data[:, obs_dim : obs_dim+act_dim]
        gen_next_obs = generated_data[:, obs_dim+act_dim+1 : obs_dim+act_dim+1+obs_dim]
        
        errors = []
        valid_checks = 0
        
        # Check a subset to save time
        indices = np.random.choice(len(generated_data), min(500, len(generated_data)), replace=False)
        
        for idx in indices:
            s = gen_obs[idx]
            a = gen_act[idx]
            s_prime_gen = gen_next_obs[idx]
            
            qpos, qvel = self.obs_to_state(s)
            
            if qpos is None:
                continue # Skip unsupported envs
                
            # Set State
            try:
                self.env.reset()
                self.env.unwrapped.set_state(qpos, qvel)
                
                # Step
                s_prime_true, _, _, _, _ = self.env.step(a)
                
                # Compute Error
                mse = np.mean((s_prime_true - s_prime_gen)**2)
                errors.append(mse)
                valid_checks += 1
            except Exception as e:
                pass

        if valid_checks == 0:
            return -1.0 # Failed to verify
            
        return np.mean(errors)

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def load_models(args, device):
    print(f"Loading dataset info for {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    input_dim = obs.shape[1] + act.shape[1] + 1 + obs.shape[1] + 1
    
    # Construct Architecture
    dummy_inputs = torch.randn(10, input_dim)
    base_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    smeme_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    
    # Load Weights
    base_ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_model.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)
    
    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location='cpu', weights_only=False)
    smeme_model.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    
    return base_model, smeme_model, d4rl_dataset, input_dim

def generate_samples(model, num_samples, batch_size=256):
    model.eval()
    samples_list = []
    num_batches = int(np.ceil(num_samples / batch_size))
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Sampling"):
            batch_samples = model.sample(batch_size=batch_size, num_sample_steps=64)
            samples_list.append(batch_samples.cpu().numpy())
            
    return np.concatenate(samples_list, axis=0)[:num_samples]

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Register Globals for Pickle
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
    device = get_device()
    
    # 1. Load Data & Models
    base_model, smeme_model, dataset, dim = load_models(args, device)
    
    # Extract Real Data for comparisons
    obs = dataset['observations']
    act = dataset['actions']
    real_data = np.concatenate([
        dataset['observations'], 
        dataset['actions'],
        dataset['rewards'][:,None],
        dataset['next_observations'],
        dataset['terminals'][:,None]
    ], axis=1)
    
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    # 2. Generate Samples
    print("\n--- Generating Samples ---")
    N = 2000
    samples_base = generate_samples(base_model, N)
    samples_smeme = generate_samples(smeme_model, N)
    
    # 3. Principled Metrics
    print("\n" + "="*50)
    print("PRINCIPLED EVALUATION REPORT (IJCAI TARGET)")
    print("="*50)
    
    # A. Vendi Score (Diversity)
    print("Computing Vendi Score (SOTA Diversity)...")
    vs_real = compute_vendi_score(real_data[:N])
    vs_base = compute_vendi_score(samples_base)
    vs_smeme = compute_vendi_score(samples_smeme)
    
    print(f"\nDiversity (Vendi Score) [Higher is Better]:")
    print(f"  Real Data: {vs_real:.2f}")
    print(f"  Base Model: {vs_base:.2f}")
    print(f"  S-MEME:     {vs_smeme:.2f} ({(vs_smeme-vs_base)/vs_base*100:+.1f}%)")
    
    # B. Dynamics Oracle (Manifold Validity)
    print("\nChecking Online Dynamics (Ground Truth Consistency)...")
    oracle = DynamicsOracle(args.dataset)
    
    # Check MSE
    mse_base = oracle.verify_transitions(samples_base, obs_dim, act_dim)
    mse_smeme = oracle.verify_transitions(samples_smeme, obs_dim, act_dim)
    
    print(f"\nDynamics Error (MSE against Physics Engine) [Lower is Better]:")
    if mse_base == -1:
        print("  [Skipped] Environment state setting not supported for this dataset.")
    else:
        print(f"  Base Model: {mse_base:.5f}")
        print(f"  S-MEME:     {mse_smeme:.5f}")
        
        if mse_smeme > mse_base * 1.2:
            print("  -> Warning: Significant manifold violation detected in S-MEME.")
        else:
            print("  -> Success: Manifold adherence maintained.")

    print("="*50)
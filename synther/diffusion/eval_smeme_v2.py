# synther/diffusion/eval_smeme_v2.py

import argparse
import torch
import numpy as np
import gymnasium as gym
import gin
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import sys

# Import Custom Modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.train_smeme import SMEMEConfig, AdjointMatchingConfig
# Use just_d4rl instead of the full d4rl package
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. PRINCIPLED METRICS: VENDI SCORE & DYNAMICS ORACLE
# ============================================================================

def compute_vendi_score(samples, kernel='rbf', gamma=None):
    """
    Computes the Vendi Score (Diversity).
    VS = exp( - sum(lambda_i * log(lambda_i)) )
    """
    if len(samples) > 5000:
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
    Doesn't require d4rl installation.
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
        """Maps d4rl dataset strings to standard Gymnasium environment IDs."""
        name = dataset_name.lower()
        if 'hopper' in name:
            return 'Hopper-v4'
        elif 'walker' in name:
            return 'Walker2d-v4'
        elif 'halfcheetah' in name:
            return 'HalfCheetah-v4'
        elif 'ant' in name:
            return 'Ant-v4'
        else:
            # Fallback or error
            return 'Hopper-v4' 

    def obs_to_state(self, obs):
        """
        Maps Observation -> MuJoCo qpos/qvel.
        """
        if self.env is None: return None, None

        # Standard MuJoCo (Hopper, Walker, Cheetah)
        # Obs = [qpos[1:], qvel]
        # We assume x-position (qpos[0]) = 0 for physics check
        try:
            model = self.env.unwrapped.model
            qpos_dim = model.nq
            qvel_dim = model.nv
            
            qpos = np.zeros(qpos_dim)
            # qpos[0] is usually x-axis (ignored in obs)
            qpos[1:] = obs[:qpos_dim-1]
            qvel = obs[qpos_dim-1:]
            return qpos, qvel
        except:
            return None, None

    def verify_transitions(self, generated_data, obs_dim, act_dim):
        """
        Computes MSE between Gen(s') and True(s, a).
        """
        if self.env is None: return -1.0

        gen_obs = generated_data[:, :obs_dim]
        gen_act = generated_data[:, obs_dim : obs_dim+act_dim]
        gen_next_obs = generated_data[:, obs_dim+act_dim+1 : obs_dim+act_dim+1+obs_dim]
        
        errors = []
        valid_checks = 0
        indices = np.random.choice(len(generated_data), min(500, len(generated_data)), replace=False)
        
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
                
                # Compare
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
    
    # Load Data (Using just_d4rl)
    print(f"Loading dataset info for {args.dataset}...")
    dataset = d4rl_offline_dataset(args.dataset)
    obs = dataset['observations']
    act = dataset['actions']
    
    # Model Setup
    input_dim = obs.shape[1] + act.shape[1] + 1 + obs.shape[1] + 1
    dummy_inputs = torch.randn(10, input_dim)
    
    base_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    smeme_model = construct_diffusion_model(inputs=dummy_inputs).to(device)
    
    # Load Weights [FIXED: Added weights_only=False]
    base_ckpt = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    base_model.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)
    
    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location='cpu', weights_only=False)
    smeme_model.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    
    # Generate
    def sample_fn(model, N):
        model.eval()
        res = []
        with torch.no_grad():
            for _ in range(int(np.ceil(N/500))):
                res.append(model.sample(batch_size=500, num_sample_steps=64).cpu().numpy())
        return np.vstack(res)[:N]

    print("\n--- Generating Samples ---")
    N = 2000
    samples_base = sample_fn(base_model, N)
    samples_smeme = sample_fn(smeme_model, N)
    
    # Real Data subset
    real_data = np.concatenate([
        dataset['observations'], 
        dataset['actions'],
        dataset['rewards'][:,None],
        dataset['next_observations'],
        dataset['terminals'][:,None]
    ], axis=1)
    
    # Evaluation
    print("\n" + "="*50)
    print("PRINCIPLED EVALUATION REPORT")
    print("="*50)
    
    # Vendi
    vs_real = compute_vendi_score(real_data[:N])
    vs_base = compute_vendi_score(samples_base)
    vs_smeme = compute_vendi_score(samples_smeme)
    
    print(f"Diversity (Vendi Score):")
    print(f"  Real: {vs_real:.2f} | Base: {vs_base:.2f} | S-MEME: {vs_smeme:.2f}")
    
    # Dynamics Oracle
    oracle = DynamicsOracle(args.dataset)
    mse_base = oracle.verify_transitions(samples_base, obs.shape[1], act.shape[1])
    mse_smeme = oracle.verify_transitions(samples_smeme, obs.shape[1], act.shape[1])
    
    print(f"\nDynamics MSE (Physics Consistency):")
    print(f"  Base Model: {mse_base:.5f}")
    print(f"  S-MEME:     {mse_smeme:.5f}")
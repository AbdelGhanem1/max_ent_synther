# synther/diffusion/eval_smeme.py

import argparse
import torch
import numpy as np
import gymnasium as gym  # Using gymnasium instead of gym
import gin  # <--- Added missing import
from tqdm import tqdm
from scipy.spatial.distance import cdist
import copy

# Import your custom modules
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.adjoint_matching_solver import AdjointMatchingSolver
from synther.diffusion.train_smeme import DiffusionModelAdapter, AdjointMatchingConfig
from just_d4rl import d4rl_offline_dataset

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. SAMPLING ENGINE
# ============================================================================

def generate_samples(model, num_samples, batch_size=256, device='cuda'):
    """
    Generates N samples from an ElucidatedDiffusion model.
    """
    model.eval()
    samples_list = []
    
    num_batches = int(np.ceil(num_samples / batch_size))
    
    print(f"Generating {num_samples} samples...")
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Sampling"):
            # Sample using the model's standard sampling procedure
            batch_samples = model.sample(
                batch_size=batch_size,
                num_sample_steps=128 # Standard for high quality
            )
            samples_list.append(batch_samples.cpu().numpy())
            
    # Concatenate and crop to exact num_samples
    all_samples = np.concatenate(samples_list, axis=0)[:num_samples]
    return all_samples

# ============================================================================
# 2. PHASE 1: STEERING CHECK (Diagnostic)
# ============================================================================

def run_steering_check(base_model, input_dim, device):
    """
    DIAGNOSTIC: Verifies that the AdjointMatchingSolver works mechanically.
    We try to maximize a simple linear reward: R(x) = mean(x).
    """
    print("\n--- Phase 1: Running Steering Check (Solver Diagnostic) ---")
    
    # 1. Setup Solver
    # Use aggressive parameters to force a visible shift quickly
    am_config = AdjointMatchingConfig(
        num_inference_steps=20,  # Fast
        reward_multiplier=100.0  # Strong signal
    )
    
    # Wrap model
    wrapped_model = DiffusionModelAdapter(base_model, am_config).to(device)
    
    # Create a temporary trainable copy to test optimization
    trainable_model = copy.deepcopy(wrapped_model)
    # We need to ensure the underlying model has gradients enabled
    for p in trainable_model.parameters():
        p.requires_grad = True
        
    optimizer = torch.optim.Adam(trainable_model.parameters(), lr=1e-4)
    
    solver = AdjointMatchingSolver(
        model_pre=wrapped_model, # Self-reference for regularization
        model_fine=trainable_model,
        config=am_config
    )
    
    # 2. Define Dummy Reward: Maximize value of index 0
    # Reward = x[:, 0].sum()
    def linear_reward_fn(x):
        return x[:, 0]
    
    print("Running 10 steps of dummy fine-tuning...")
    initial_loss = None
    
    for step in range(10):
        optimizer.zero_grad()
        noise = torch.randn(128, input_dim, device=device)
        
        # This solves for the control that maximizes the linear reward
        loss = solver.solve(noise, linear_reward_fn)
        loss.backward()
        optimizer.step()
        
        if initial_loss is None:
            initial_loss = loss.item()
            
        print(f"  Step {step}: Loss = {loss.item():.4f}")

    if loss.item() < initial_loss:
        print("Steering Check: PASSED (Loss decreased)")
    else:
        print("Steering Check: WARNING (Loss did not decrease, check hyperparameters)")
    
    return True

# ============================================================================
# 3. PHASE 2: EXPANSION METRICS (Entropy Proxies)
# ============================================================================

def compute_pairwise_diversity(samples, subsample=1000):
    """
    Metric: Average Euclidean distance between pairs of samples.
    """
    indices = np.random.choice(len(samples), min(len(samples), subsample), replace=False)
    subset = samples[indices]
    
    dists = cdist(subset, subset, metric='euclidean')
    
    # Exclude diagonal (distance to self is 0)
    mask = np.ones_like(dists, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    avg_dist = dists[mask].mean()
    return avg_dist

def compute_novelty(generated_samples, training_samples, subsample_gen=1000, subsample_train=10000):
    """
    Metric: Average distance to the NEAREST neighbor in the training set.
    """
    gen_indices = np.random.choice(len(generated_samples), min(len(generated_samples), subsample_gen), replace=False)
    # Ensure we don't sample more than available
    n_train = min(len(training_samples), subsample_train)
    train_indices = np.random.choice(len(training_samples), n_train, replace=False)
    
    gen_subset = generated_samples[gen_indices]
    train_subset = training_samples[train_indices]
    
    dists = cdist(gen_subset, train_subset, metric='euclidean')
    min_dists = dists.min(axis=1)
    
    return min_dists.mean()

# ============================================================================
# 4. PHASE 3: MANIFOLD VALIDITY (Kinematic Checks)
# ============================================================================

def evaluate_kinematics(gen_data, real_data, obs_dim, act_dim):
    """
    Computes distribution shift in "Velocity" (s' - s).
    If the model generates teleportation, this metric will explode.
    """
    # Extract s and s'
    # Structure: [obs (N), act (M), rew (1), next_obs (N), term (1)]
    
    gen_s = gen_data[:, :obs_dim]
    gen_next_s = gen_data[:, obs_dim+act_dim+1 : obs_dim+act_dim+1+obs_dim]
    
    real_s = real_data[:, :obs_dim]
    real_next_s = real_data[:, obs_dim+act_dim+1 : obs_dim+act_dim+1+obs_dim]
    
    # Compute Deltas (Approximate Velocities)
    gen_delta = gen_next_s - gen_s
    real_delta = real_next_s - real_s
    
    # 1. Action Validity Check
    # Actions should be roughly in [-1, 1] for D4RL
    gen_a = gen_data[:, obs_dim : obs_dim+act_dim]
    act_violation = (np.abs(gen_a) > 1.05).mean() * 100 # % of violations
    
    # 2. Kinematic Bound Check
    # Do generated deltas exceed the max delta seen in dataset?
    # We look at the 99th percentile to be robust to dataset outliers
    real_max_delta = np.percentile(np.abs(real_delta), 99, axis=0)
    # Avoid division by zero
    real_max_delta = np.maximum(real_max_delta, 1e-6)
    
    gen_max_delta = np.percentile(np.abs(gen_delta), 99, axis=0)
    
    # Ratio: How much faster are we moving than physically observed?
    # Ratio > 1.5 implies potential physics violation
    kinematic_violation_ratio = np.mean(gen_max_delta / real_max_delta)
    
    return act_violation, kinematic_violation_ratio

# ============================================================================
# INFRASTRUCTURE SETUP
# ============================================================================

def load_models(args, device):
    """
    Loads Base Model and S-MEME Finetuned Model.
    """
    print(f"Loading dataset info for {args.dataset}...")
    d4rl_dataset = d4rl_offline_dataset(args.dataset)
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    
    # Concatenate to get input dim (obs + act + r + s' + t)
    # Assuming standard structure dimensions
    input_dim = obs.shape[1] + act.shape[1] + 1 + obs.shape[1] + 1
    
    # 2. Construct Architecture
    # Use dummy inputs with batch size > 1 to avoid std() NaN issues during init
    dummy_inputs = torch.randn(10, input_dim)
    
    print("Constructing Base Model...")
    base_model = construct_diffusion_model(inputs=dummy_inputs)
    print("Constructing S-MEME Model...")
    smeme_model = construct_diffusion_model(inputs=dummy_inputs)
    
    # 3. Load Weights
    print(f"Loading Base Model from {args.base_checkpoint}...")
    base_ckpt = torch.load(args.base_checkpoint, map_location='cpu')
    base_model.load_state_dict(base_ckpt['model'] if 'model' in base_ckpt else base_ckpt)
    base_model.to(device)
    
    print(f"Loading S-MEME Model from {args.smeme_checkpoint}...")
    smeme_ckpt = torch.load(args.smeme_checkpoint, map_location='cpu')
    smeme_model.load_state_dict(smeme_ckpt['model'] if 'model' in smeme_ckpt else smeme_ckpt)
    smeme_model.to(device)
    
    return base_model, smeme_model, d4rl_dataset, input_dim

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="D4RL environment name")
    parser.add_argument('--base_checkpoint', type=str, required=True, help="Original Synther model")
    parser.add_argument('--smeme_checkpoint', type=str, required=True, help="S-MEME finetuned model")
    parser.add_argument('--num_eval_samples', type=int, default=5000)
    parser.add_argument('--skip_steering', action='store_true', help="Skip the diagnostic training phase")
    # GIN args required by construct_diffusion_model
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    
    args = parser.parse_args()
    
    # Initialize GIN
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    
    device = get_device()
    print(f"Running S-MEME Evaluation on {device}...")
    
    # 1. Load Models & Data
    base_model, smeme_model, d4rl_dataset, input_dim = load_models(args, device)
    
    # Reconstruct Dataset numpy array for comparisons
    obs = d4rl_dataset['observations']
    act = d4rl_dataset['actions']
    rew = d4rl_dataset['rewards'][:, None] if len(d4rl_dataset['rewards'].shape)==1 else d4rl_dataset['rewards']
    term = d4rl_dataset['terminals'][:, None] if len(d4rl_dataset['terminals'].shape)==1 else d4rl_dataset['terminals']
    next_obs = d4rl_dataset['next_observations']
    
    real_data = np.concatenate([obs, act, rew, next_obs, term], axis=1)
    
    obs_dim = obs.shape[1]
    act_dim = act.shape[1]
    
    # 2. Phase 1: Steering Check
    if not args.skip_steering:
        print("\n--- PHASE 1: DIAGNOSTIC ---")
        # Pass a deep copy to avoid modifying the eval model
        diag_model = copy.deepcopy(base_model)
        run_steering_check(diag_model, input_dim, device)
        del diag_model
    
    # 3. Generate Samples
    print(f"\n--- PHASE 2: GENERATION ({args.num_eval_samples} samples) ---")
    print("1. Sampling Base Model...")
    base_samples = generate_samples(base_model, args.num_eval_samples, device=device)
    
    print("2. Sampling S-MEME Model...")
    smeme_samples = generate_samples(smeme_model, args.num_eval_samples, device=device)
    
    # 4. Compute Metrics
    print("\n--- PHASE 3: METRICS ---")
    
    # Diversity
    div_base = compute_pairwise_diversity(base_samples)
    div_smeme = compute_pairwise_diversity(smeme_samples)
    div_delta = ((div_smeme - div_base) / div_base) * 100
    
    # Novelty
    nov_base = compute_novelty(base_samples, real_data)
    nov_smeme = compute_novelty(smeme_samples, real_data)
    nov_delta = ((nov_smeme - nov_base) / nov_base) * 100
    
    # Kinematics
    act_viol_base, kin_ratio_base = evaluate_kinematics(base_samples, real_data, obs_dim, act_dim)
    act_viol_smeme, kin_ratio_smeme = evaluate_kinematics(smeme_samples, real_data, obs_dim, act_dim)
    
    # 5. Final Report
    print("\n" + "="*60)
    print(f"S-MEME EVALUATION REPORT: {args.dataset}")
    print("="*60)
    print(f"{'Metric':<25} | {'Base Model':<12} | {'S-MEME':<12} | {'Delta':<10}")
    print("-" * 65)
    print(f"{'Diversity (Spread)':<25} | {div_base:.4f}       | {div_smeme:.4f}       | {div_delta:+.1f}%")
    print(f"{'Novelty (Gap Filling)':<25} | {nov_base:.4f}       | {nov_smeme:.4f}       | {nov_delta:+.1f}%")
    print("-" * 65)
    print(f"{'Action Violations (>1)':<25} | {act_viol_base:.2f}%       | {act_viol_smeme:.2f}%       | {(act_viol_smeme-act_viol_base):+.2f}%")
    print(f"{'Kinematic Ratio':<25} | {kin_ratio_base:.2f}x        | {kin_ratio_smeme:.2f}x        | {(kin_ratio_smeme-kin_ratio_base):+.2f}")
    print("="*60)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if div_delta > 1.0:
        print(f"[SUCCESS] S-MEME increased sample diversity by {div_delta:.1f}%.")
    else:
        print("[WARNING] S-MEME failed to expand the distribution (check alpha/step size).")
        
    if kin_ratio_smeme > 1.5:
        print("[FAILURE] S-MEME broke physics constraints (Kinematic Ratio > 1.5x).")
        print("          The model is generating impossible transitions (teleportation).")
    elif kin_ratio_smeme > 1.1:
        print(f"[WARNING] Slight kinematic drift detected ({kin_ratio_smeme:.2f}x).")
    else:
        print("[SUCCESS] Physical validity maintained (Kinematics match dataset).")
        
    print("="*60)
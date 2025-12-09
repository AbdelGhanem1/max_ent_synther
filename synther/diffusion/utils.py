# Utilities for diffusion.
from typing import Optional, List, Union

# [MODIFIED] Replaced d4rl and gym imports
# import d4rl
# import gym
import gymnasium as gym
from just_d4rl import d4rl_offline_dataset

import gin
import numpy as np
import torch
from torch import nn

# GIN-required Imports.
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory


# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    # [MODIFIED] Use just_d4rl instead of d4rl.qlearning_dataset(env)
    # Note: env.spec.id retrieves the string name (e.g., 'hopper-medium-v2')
    dataset = d4rl_offline_dataset(env.spec.id)
    
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    
    # just_d4rl returns (N,) for rewards, we need (N, 1)
    if len(rewards.shape) == 1:
        rewards = rewards[:, None]
        
    inputs = np.concatenate([obs, actions, rewards, next_obs], axis=1)
    
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        if len(terminals.shape) == 1:
            terminals = terminals[:, None]
        inputs = np.concatenate([inputs, terminals], axis=1)
        
    return inputs


# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    # [NOTE] Gymnasium envs function the same way for these attributes
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )
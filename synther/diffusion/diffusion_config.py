# FILE: synther/diffusion/diffusion_config.py
from dataclasses import dataclass

@dataclass
class EDMConfig:
    """
    The Single Source of Truth for Diffusion Physics.
    These match Karras et al. (2022) / Elucidated Diffusion defaults.
    """
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 1.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2

# Global instance to be imported everywhere
edm_global_config = EDMConfig()
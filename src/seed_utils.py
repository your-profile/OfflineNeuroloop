"""Seeds the random number generators for each experiment"""
from __future__ import annotations
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def begin_rl_training(trial_seed: int) -> int:
    """Reset global RNGs for the RL phase"""
    set_global_seed(trial_seed)
    return int(np.random.randint(0, 1000000))

import random
import numpy as np
import torch

__all__ = ["set_random_seed"]

def set_random_seed(seed: int = 42):
    """Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Configures the following sources of randomness:

    - Python's built-in ``random`` module
    - NumPy's global random state
    - PyTorch CPU random generator
    - PyTorch CUDA random generators (all GPUs) if CUDA is available

    Also sets ``torch.backends.cudnn.deterministic = True`` and
    ``torch.backends.cudnn.benchmark = False`` to disable non-deterministic
    cuDNN algorithms.

    Args:
        seed (int): Seed value to use. Defaults to ``42``.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed: {seed}")

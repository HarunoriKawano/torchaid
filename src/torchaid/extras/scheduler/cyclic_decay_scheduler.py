import torch
from torch.optim.lr_scheduler import CyclicLR

__all__ = ["get_cycle_scheduler"]

def get_cycle_scheduler(optimizer: torch.optim.Optimizer, max_lr: float, min_lr: float, warmup_steps: int, down_steps: int):
    """Creates a triangular2 cyclic learning rate scheduler.

    Uses :class:`torch.optim.lr_scheduler.CyclicLR` in ``"triangular2"`` mode,
    which halves the amplitude of each cycle, causing the learning rate to
    converge over time.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will
            be cycled.
        max_lr (float): Maximum learning rate (peak of each cycle).
        min_lr (float): Minimum learning rate (base of each cycle).
        warmup_steps (int): Number of steps to increase the learning rate from
            ``min_lr`` to ``max_lr`` (``step_size_up``).
        down_steps (int): Number of steps to decrease the learning rate from
            ``max_lr`` back to ``min_lr`` (``step_size_down``).

    Returns:
        CyclicLR: A configured :class:`~torch.optim.lr_scheduler.CyclicLR`
            scheduler in ``"triangular2"`` mode.
    """
    if max_lr <= 0:
        raise ValueError(f"max_lr must be positive, got {max_lr}")
    if min_lr < 0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}")
    if min_lr >= max_lr:
        raise ValueError(f"min_lr ({min_lr}) must be less than max_lr ({max_lr})")
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be a positive integer, got {warmup_steps}")
    if down_steps <= 0:
        raise ValueError(f"down_steps must be a positive integer, got {down_steps}")

    return CyclicLR(optimizer, min_lr, max_lr, step_size_up=warmup_steps, step_size_down=down_steps, mode="triangular2")

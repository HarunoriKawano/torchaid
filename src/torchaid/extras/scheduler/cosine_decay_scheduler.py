import math

import torch
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["get_cosine_scheduler"]

class CosineDecayScheduler:
    """Learning rate multiplier with linear warm-up followed by cosine decay.

    Used as the ``lr_lambda`` callable for :class:`torch.optim.lr_scheduler.LambdaLR`.
    During the warm-up phase the multiplier increases linearly from ``0`` to ``1``.
    After warm-up it follows a cosine curve from ``1`` down to ``min_weight``.

    Attributes:
        _warmup_steps (int): Effective number of warm-up steps (capped at
            ``max_warmup_steps``).
        _max_steps (int): Total number of training steps.
        _min_weight (float): Minimum learning rate multiplier after decay.
    """

    def __init__(self, warmup_steps: int, max_steps: int,
                 max_warmup_steps: int = 10000, min_weight: float = 0.05):
        """Initializes the cosine decay scheduler.

        Args:
            warmup_steps (int): Desired number of linear warm-up steps.
            max_steps (int): Total number of training steps (warm-up + decay).
            max_warmup_steps (int): Upper bound on warm-up steps. If
                ``warmup_steps > max_warmup_steps``, the effective warm-up is
                capped at ``max_warmup_steps``. Defaults to ``10000``.
            min_weight (float): Floor value for the cosine decay, preventing the
                learning rate from dropping below ``min_weight * base_lr``.
                Defaults to ``0.05``.
        """
        if max_steps <= 0:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps > max_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) must not exceed max_steps ({max_steps})")
        if not 0.0 <= min_weight <= 1.0:
            raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")

        if max_warmup_steps < warmup_steps:
            self._warmup_steps = max_warmup_steps
        else:
            self._warmup_steps = warmup_steps
        self._max_steps = max_steps
        self._min_weight = min_weight

    def __call__(self, epoch: int):
        """Computes the learning rate multiplier for a given step.

        Args:
            epoch (int): Current training step (0-indexed as passed by
                :class:`~torch.optim.lr_scheduler.LambdaLR`).

        Returns:
            float: Learning rate multiplier in ``[min_weight, 1.0]``.
                During warm-up: ``epoch / warmup_steps``.
                After warm-up: cosine decay to ``min_weight``.
        """
        epoch = max(epoch, 1)
        if epoch <= self._warmup_steps:
            return epoch / self._warmup_steps
        epoch -= 1
        rad = math.pi * (epoch - self._warmup_steps) / (self._max_steps - self._warmup_steps)
        weight = (math.cos(rad) + 1.) / 2

        return max(weight, self._min_weight)


def get_cosine_scheduler(
        optimizer: torch.optim.Optimizer, warmup_steps: int, max_steps: int,
        max_warmup_steps: int = 10000, min_weight: float = 0.05
):
    """Creates a LambdaLR scheduler with linear warm-up and cosine decay.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will
            be scaled.
        warmup_steps (int): Number of linear warm-up steps.
        max_steps (int): Total number of training steps.
        max_warmup_steps (int): Maximum allowed warm-up steps. Defaults to
            ``10000``.
        min_weight (float): Minimum learning rate multiplier at the end of
            decay. Defaults to ``0.05``.

    Returns:
        LambdaLR: A :class:`~torch.optim.lr_scheduler.LambdaLR` scheduler
            backed by :class:`CosineDecayScheduler`.
    """
    return LambdaLR(optimizer,
                    CosineDecayScheduler(warmup_steps, max_steps, max_warmup_steps, min_weight)
                    )

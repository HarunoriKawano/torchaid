from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import nn

from .configs import Mode

__all__ = ['TaskModule']

class TaskModule(nn.Module, ABC):
    """Abstract base class for task-specific PyTorch models.

    Combines :class:`torch.nn.Module` with the :class:`~torchaid.core.configs.Mode`-aware
    forward interface required by :class:`~torchaid.core.trainer.TrainFramework`.
    Concrete subclasses implement the task logic inside :meth:`forward`.
    """

    @abstractmethod
    def forward(self, mode: Mode, batch: dict[str, Any]) -> tuple[dict[str, Any], Optional[Any]]:
        """Runs the forward pass for the given operation mode.

        Args:
            mode (Mode): Indicates whether the call originates from a training,
                validation, or test step. Implementations may return different
                output keys depending on the mode (e.g., omitting predictions
                during training).
            batch (dict[str, Any]): Task-specific input data for the current batch.

        Returns:
            tuple[dict[str, Any], Optional[Any]]: A 2-tuple of ``(outputs, error)``.
            On success, ``error`` must be ``None`` and ``outputs`` must include a
            scalar ``"loss"`` tensor for training steps. To signal a recoverable
            per-batch error, return a non-``None`` value as ``error``; the framework
            will skip backpropagation and log it to ``stderr``.
        """
        pass

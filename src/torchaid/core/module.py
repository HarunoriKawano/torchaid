from abc import ABC, abstractmethod

from torch import nn

from . import BaseInputs, BaseOutputs
from .configs import Mode

__all__ = ['TaskModule']

class TaskModule(nn.Module, ABC):
    """Abstract base class for task-specific PyTorch models.

    Combines :class:`torch.nn.Module` with the :class:`~torchaid.core.configs.Mode`-aware
    forward interface required by :class:`~torchaid.core.trainer.TrainFramework`.
    Concrete subclasses implement the task logic inside :meth:`forward`.
    """

    @abstractmethod
    def forward(self, mode: Mode, batch: BaseInputs) -> BaseOutputs:
        """Runs the forward pass for the given operation mode.

        Args:
            mode (Mode): Indicates whether the call originates from a training,
                validation, or test step. Implementations may return different
                :class:`~torchaid.core.configs.BaseOutputs` subclasses depending
                on the mode (e.g., omitting predictions during training).
            batch (BaseInputs): Task-specific input data for the current batch.

        Returns:
            BaseOutputs: Model outputs including at least a scalar ``loss`` tensor.
                Additional fields (e.g., predictions) may be included depending
                on the mode and task.
        """
        pass

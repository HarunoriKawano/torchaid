from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

from . import BaseInputs, BaseOutputs

__all__ = ['BaseMetricCalculator']
T = TypeVar('T', bound='BaseMetrics')

class BaseMetricCalculator(ABC, Generic[T]):
    """Abstract base class for computing and managing training metrics.

    Concrete subclasses implement per-step metric accumulation logic and
    epoch-level aggregation for training, validation, and test phases.

    Type Parameters:
        T: A :class:`~torchaid.core.configs.BaseMetrics` subclass that holds
           the metric state for this task.

    Attributes:
        metrics (T): Mutable metrics object that accumulates values across steps
            and epochs.
    """

    def __init__(self, metrics: T):
        """Initializes the calculator with a metrics container.

        Args:
            metrics (T): An instance of a :class:`~torchaid.core.configs.BaseMetrics`
                subclass used to store accumulated metric values.
        """
        self.metrics = metrics

    @abstractmethod
    def train_step(self, outputs: BaseOutputs, batch: BaseInputs) -> dict[str, Any]:
        """Processes a single training step and updates internal accumulators.

        Args:
            outputs (BaseOutputs): Model outputs returned by the forward pass.
            batch (BaseInputs): Input batch used for the current step.

        Returns:
            dict[str, Any]: Key-value pairs to display in the progress bar
                (e.g., ``{"loss": 0.342}``).
        """
        pass

    @abstractmethod
    def val_step(self, outputs: BaseOutputs, batch: BaseInputs) -> dict[str, Any]:
        """Processes a single validation step and updates internal accumulators.

        Args:
            outputs (BaseOutputs): Model outputs returned by the forward pass.
            batch (BaseInputs): Input batch used for the current step.

        Returns:
            dict[str, Any]: Key-value pairs to display in the progress bar.
        """
        pass

    @abstractmethod
    def test_step(self, outputs: BaseOutputs, batch: BaseInputs) -> dict[str, Any]:
        """Processes a single test step and updates internal accumulators.

        Args:
            outputs (BaseOutputs): Model outputs returned by the forward pass.
            batch (BaseInputs): Input batch used for the current step.

        Returns:
            dict[str, Any]: Key-value pairs to display in the progress bar.
        """
        pass

    @abstractmethod
    def check(self) -> bool:
        """Aggregates accumulated values into epoch-level metrics and checks for improvement.

        Called at the end of each epoch after validation. Subclasses should
        compute summary statistics (e.g., mean loss, accuracy) and store them
        in :attr:`metrics`.

        Returns:
            bool: ``True`` if the model has improved compared to the previous
                best result (e.g., lower validation loss), ``False`` otherwise.
        """
        pass

    @abstractmethod
    def test(self):
        """Finalizes test-phase metrics after all test steps have been processed.

        Called once after the test loop completes. Subclasses should compute
        and store final test statistics in :attr:`metrics`.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clears all per-epoch accumulators in preparation for the next epoch.

        Called at the beginning of each epoch before the training loop starts.
        """
        pass

    def system_check(self):
        """Prints the current metrics state to stdout.

        Used during the system sanity check phase before full training begins.
        """
        print(self.metrics)

    def early_stopping(self) -> bool:
        """Determines whether training should stop early.

        The default implementation never triggers early stopping. Subclasses
        may override this method to implement patience-based or
        metric-threshold-based stopping criteria.

        Returns:
            bool: ``True`` to stop training, ``False`` to continue.
                Always returns ``False`` in the base implementation.
        """
        return False

    def replace(self, data: dict[str, Any]):
        """Updates metric fields from a dictionary, typically loaded from a checkpoint.

        Only keys that correspond to declared fields of :attr:`metrics` are applied;
        unrecognized keys are silently ignored.

        Args:
            data (dict[str, Any]): Mapping of field names to values to restore.
        """
        for key, val in data.items():
            if key in self.metrics.model_fields:
                setattr(self.metrics, key, val)

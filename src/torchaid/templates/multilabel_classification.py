import math
from typing import Optional, Any
import statistics

from sklearn.metrics import f1_score
import torch
from torch import nn

from torchaid import TaskModule, Mode, BaseMetrics, BaseMetricCalculator

__all__ = ["MultiLabelClassification"]


class MultiLabelClassification(TaskModule):
    """Task module for multi-class classification using cross-entropy loss.

    Wraps an arbitrary feature-extraction backbone and adds a
    :class:`torch.nn.CrossEntropyLoss` criterion on top.

    Attributes:
        model (nn.Module): Backbone network that maps input tensors to logits.
        criteria (nn.CrossEntropyLoss): Loss function.
    """

    def __init__(self, model: nn.Module):
        """Initializes the classification module.

        Args:
            model (nn.Module): Backbone network. Must accept the ``inputs``
                tensor and return logits of shape ``(B, num_classes)``.
        """
        super().__init__()
        self.model = model
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, mode: Mode, batch: dict[str, Any]) -> tuple[dict[str, Any], Optional[Any]]:
        """Runs the forward pass and computes the cross-entropy loss.

        During training only the loss is returned. During validation and
        testing the predicted class indices are also included.

        Args:
            mode (Mode): Operation mode. Returns :class:`TrainOutputs` when
                ``mode`` is :attr:`~torchaid.core.configs.Mode.TRAIN`, otherwise
                returns :class:`EvalOutputs`.
            batch (Inputs): Input batch containing feature tensors and labels.

        Returns:
            BaseOutputs: :class:`TrainOutputs` in training mode, or
                :class:`EvalOutputs` (with ``predicted`` field) otherwise.
        """
        out = self.model(batch["inputs"])
        loss = self.criteria(out, batch["labels"])
        if mode == Mode.TRAIN:
            return {"loss": loss}, None

        predicted = out.argmax(dim=-1)
        return {"loss": loss, "predicted": predicted}, None

class Metrics(BaseMetrics):
    """Metrics container for multi-label classification.

    Attributes:
        train_loss (Optional[float]): Mean training loss for the current epoch.
        val_loss (Optional[float]): Mean validation loss for the current epoch.
        val_accuracy (Optional[float]): Validation accuracy for the current epoch.
        val_f1_score (Optional[float]): Macro-averaged F1 score on the validation set.
        best_val_loss (float): Best (lowest) validation loss observed so far.
        best_epoch (int): Epoch at which the best validation loss was achieved.
        test_loss (Optional[float]): Mean loss on the test set.
        test_accuracy (Optional[float]): Accuracy on the test set.
        test_f1_score (Optional[float]): Macro-averaged F1 score on the test set.
    """

    train_loss: Optional[float] = None

    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_f1_score: Optional[float] = None

    best_val_loss: float = math.inf
    best_epoch: int = 0

    test_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    test_f1_score: Optional[float] = None

class MetricsCalculator(BaseMetricCalculator[Metrics]):
    """Metric calculator for multi-label classification tasks.

    Accumulates per-step losses and predictions, then computes epoch-level
    accuracy, macro F1 score, and best-model tracking at the end of each epoch.
    """

    def __init__(self):
        """Initializes the calculator with an empty :class:`Metrics` instance."""
        super().__init__(Metrics())
        self._train_loss_list: list[float] = []
        self._eval_loss_list: list[float] = []
        self._predicted: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []

    def train_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Records the loss for a single training step.

        Args:
            outputs (TrainOutputs): Outputs from the training forward pass.
            batch (Inputs): Input batch for the current step.

        Returns:
            dict[str, Any]: Dictionary ``{"loss": <float>}`` for progress bar display.
        """
        loss = outputs["loss"].item()
        self._train_loss_list.append(loss)
        return {"loss": loss}

    def val_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Records the loss and predictions for a single validation step.

        Args:
            outputs (EvalOutputs): Outputs from the evaluation forward pass,
                including predicted class indices.
            batch (Inputs): Input batch containing ground-truth labels.

        Returns:
            dict[str, Any]: Dictionary ``{"loss": <float>}`` for progress bar display.
        """
        loss = outputs["loss"].item()
        self._eval_loss_list.append(loss)
        self._predicted.append(outputs["predicted"])
        self._labels.append(batch["labels"])

        return {"loss": loss}

    def test_step(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
        """Records metrics for a single test step (delegates to :meth:`val_step`).

        Args:
            outputs (EvalOutputs): Outputs from the test forward pass.
            batch (Inputs): Input batch containing ground-truth labels.

        Returns:
            dict[str, Any]: Dictionary ``{"loss": <float>}`` for progress bar display.
        """
        return self.val_step(outputs, batch)

    def eval(self):
        """Computes accuracy and macro F1 score from accumulated predictions.

        Concatenates all stored predictions and labels, then computes
        overall classification accuracy and macro-averaged F1 score using
        ``sklearn.metrics.f1_score``.

        Returns:
            tuple[float, float]: A ``(accuracy, f1_macro)`` pair where both
                values are in the range ``[0, 1]``.
        """
        predicted = torch.cat(self._predicted, dim=0)
        labels = torch.cat(self._labels, dim=0)
        accuracy = torch.as_tensor(predicted == labels).sum().item() / len(predicted)

        y_pred = predicted.cpu().numpy()
        y_true = labels.cpu().numpy()
        f1_macro = f1_score(y_true, y_pred, average='macro')

        return accuracy, f1_macro


    def check(self) -> bool:
        """Aggregates epoch metrics and checks whether the model has improved.

        Computes mean training loss, mean validation loss, accuracy, and macro F1
        for the completed epoch. Updates ``best_val_loss`` and ``best_epoch``
        if the current validation loss is a new minimum.

        Returns:
            bool: ``True`` if validation loss improved, ``False`` otherwise.
        """
        self.metrics.train_loss = statistics.mean(self._train_loss_list)
        self.metrics.val_loss = statistics.mean(self._eval_loss_list)
        self.metrics.val_accuracy, self.metrics.val_f1_score = self.eval()

        if self.metrics.val_loss < self.metrics.best_val_loss:
            self.metrics.best_val_loss = self.metrics.val_loss
            self.metrics.best_epoch = self.metrics.epoch
            return True
        return False

    def test(self):
        """Finalizes test metrics from accumulated predictions and losses.

        Stores mean test loss, accuracy, and macro F1 into :attr:`metrics`.
        """
        self.metrics.test_loss = statistics.mean(self._eval_loss_list)
        self.metrics.test_accuracy, self.metrics.test_f1_score = self.eval()

    def reset(self):
        """Clears all per-epoch accumulators."""
        self._train_loss_list.clear()
        self._eval_loss_list.clear()
        self._predicted.clear()
        self._labels.clear()

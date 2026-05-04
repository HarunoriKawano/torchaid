from abc import ABC
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

__all__ = ['BaseMetrics', 'BaseSettings', 'Mode']

class BaseMetrics(BaseModel, ABC):
    """Abstract base class for tracking training metrics.

    Subclasses extend this with task-specific metric fields.
    The ``step`` and ``epoch`` counters are managed automatically by
    :class:`~torchaid.core.trainer.TrainFramework`.

    Attributes:
        step (int): Total number of training steps completed so far.
        epoch (int): Total number of epochs completed so far.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    step: int = 0
    epoch: int = 0

class BaseSettings(BaseModel, ABC):
    """Abstract base class for training configuration.

    Subclasses add task-specific hyper-parameters on top of the common
    fields defined here.

    Attributes:
        batch_size (int): Number of samples per mini-batch.
        max_epoch_num (int): Maximum number of training epochs.
        mixed_precision (bool): Whether to use automatic mixed-precision training.
            Defaults to ``False``.
        precision_dtype (Literal["float16", "bfloat16"]): Floating-point dtype used
            when ``mixed_precision`` is enabled. Defaults to ``"bfloat16"``.
        cpu_num_works (int): Number of worker processes for ``DataLoader``.
            Defaults to ``4``.
        device (Literal["cuda", "cpu"]): Device on which the model is trained.
            Automatically set to ``"cuda"`` when a GPU is available, otherwise
            falls back to ``"cpu"``.
    """

    batch_size: Annotated[int, Field(gt=0)]
    max_epoch_num: Annotated[int, Field(gt=0)]
    mixed_precision: bool = False
    precision_dtype: Literal["float16", "bfloat16"] = "bfloat16"
    cpu_num_works: Annotated[int, Field(ge=0)] = 4
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"


class Mode(Enum):
    """Enumeration of operation modes used during the training pipeline.

    Attributes:
        TRAIN: Forward pass with gradient computation and parameter updates.
        VAL: Forward pass without gradient computation, used for validation.
        TEST: Forward pass without gradient computation, used for final evaluation.
    """

    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"

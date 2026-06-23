from typing import Any
from abc import ABC, abstractmethod

from torch import nn

from .states import BatchState, FitContext

__all__ = ['TaskModuleInterface']

class TaskModuleInterface(nn.Module, ABC):

    @abstractmethod
    def train_step(self, batch_state: BatchState, fit_context: FitContext) -> BatchState:...

    @abstractmethod
    def val_step(self, batch_state: BatchState) -> BatchState:...

    def test_step(self, batch_state: BatchState) -> BatchState:
        return self.val_step(batch_state)

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:...

    @abstractmethod
    def save(self, save_dir: str) -> None: ...

    @abstractmethod
    def load(self, save_dir: str) -> None: ...

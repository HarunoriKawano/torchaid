from abc import ABC, abstractmethod
from typing import Optional, Iterable

from pydantic import BaseModel
import torch
from pydantic import computed_field
from torch.optim.lr_scheduler import LRScheduler

from .protocols import SizedIterable
from .configs import HyperParameters
from .metrics import MetricInterface

class GlobalState(BaseModel, ABC):
    best_metric: float
    patience_count: int = 0
    current_epoch: int = 0
    global_step: int = 0

    def one_step(self) -> None:
        self.global_step += 1

    def one_epoch(self) -> None:
        self.current_epoch += 1

    def check_metric(self, metric: MetricInterface) -> bool:
        better_result = self._metric_update(metric)
        if better_result:
            self.patience_count = 0
        else:
            self.patience_count += 1

        return better_result

    def save(self, path: str) -> None:
        json_str = self.model_dump_json()
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    @classmethod
    def load(cls, path: str) -> "GlobalState":
        with open(path, "r", encoding="utf-8") as f:
            json_data = f.read()
        return cls.model_validate_json(json_data)

    @abstractmethod
    def _metric_update(self, metric: MetricInterface) -> bool: ...

class BatchState(BaseModel, ABC):
    loss: Optional[torch.Tensor] = None

    @abstractmethod
    def to(self, device: torch.device): ...


class FitContext(BaseModel, ABC):
    train_dataloader: SizedIterable[BatchState]
    val_dataloader: SizedIterable[BatchState]
    global_state: GlobalState
    hyper_parameters: HyperParameters
    current_step: int = 0
    on_fit: bool = True
    scheduler: Optional[LRScheduler] = None

    @computed_field
    @property
    def total_steps(self) -> int:
        return len(self.train_dataloader) * (self.hyper_parameters.max_epoch - self.global_state.current_epoch + 1)

    @computed_field
    @property
    def remaining_epoch(self) -> int:
        return self.hyper_parameters.max_epoch - self.global_state.current_epoch

    @computed_field
    @property
    def grad_scaler(self) -> Optional[torch.amp.GradScaler]:
        if self.hyper_parameters.amp: return torch.amp.GradScaler()
        return None

    def one_step(self) -> None:
        self.current_step += 1
        self.global_state.one_step()

    def one_epoch(self) -> None:
        self.current_step = 0
        self.global_state.one_epoch()


from abc import ABC, abstractmethod
from typing import Literal

from .states import BatchState

__all__ = ['MetricInterface']

class MetricInterface(ABC):
    """
    評価指標を計算するためのインターフェース。
    バッチごとに状態を蓄積し、エポックの最後に計算・リセットする。
    """
    @abstractmethod
    def update(self, batch_state: BatchState, mode: Literal["val", "test"]) -> None: ...

    @abstractmethod
    def compute(self, mode: Literal["val", "test"]) -> None:
        """蓄積された状態から最終的な指標を計算して返す"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """内部状態を初期化する（次のエポック用）"""
        ...
from abc import ABC

from .configs import CoreComponents
from .states import BatchState, FitContext

class CallbackInterface(ABC):
    """
    学習ループの各フックポイントで呼ばれる基底クラス。
    必要なメソッドだけを子クラスでオーバーライドして使用する。
    """
    def on_fit_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...
    def on_fit_end(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...

    def on_train_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...
    def on_train_end(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...
    def on_train_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None: ...
    def on_train_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None: ...

    def on_val_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...
    def on_val_end(self, core_components: CoreComponents, fit_context: FitContext) -> None: ...
    def on_val_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None: ...
    def on_val_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None: ...

    def on_test_begin(self, core_components: CoreComponents) -> None: ...
    def on_test_end(self, core_components: CoreComponents) -> None: ...
    def on_test_batch_start(self, core_components: CoreComponents, batch_state: BatchState) -> None: ...
    def on_test_batch_end(self, core_components: CoreComponents, batch_state: BatchState) -> None: ...


class CallbackManager(CallbackInterface):
    def __init__(self, callbacks: list[CallbackInterface]):
        self._callbacks = callbacks

    def on_fit_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_fit_begin(core_components, fit_context)

    def on_fit_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_fit_end(core_components, fit_context)

    def on_train_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_train_begin(core_components, fit_context)

    def on_train_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_train_end(core_components, fit_context)

    def on_train_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_train_batch_start(core_components, fit_context, batch_state)

    def on_train_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_train_batch_end(core_components, fit_context, batch_state)

    def on_val_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_val_begin(core_components, fit_context)

    def on_val_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        for callback in self._callbacks:
            callback.on_val_end(core_components, fit_context)

    def on_val_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_val_batch_start(core_components, fit_context, batch_state)

    def on_val_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_val_batch_end(core_components, fit_context, batch_state)

    def on_test_begin(self, core_components: CoreComponents) -> None:
        for callback in self._callbacks:
            callback.on_test_begin(core_components)

    def on_test_end(self, core_components: CoreComponents) -> None:
        for callback in self._callbacks:
            callback.on_test_end(core_components)

    def on_test_batch_start(self, core_components: CoreComponents, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_test_batch_start(core_components, batch_state)

    def on_test_batch_end(self, core_components: CoreComponents, batch_state: BatchState) -> None:
        for callback in self._callbacks:
            callback.on_test_batch_end(core_components, batch_state)
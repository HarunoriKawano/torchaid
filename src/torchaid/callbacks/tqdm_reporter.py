from typing import Any

from tqdm import tqdm

from torchaid.core.callback import CallbackInterface
from torchaid.core.configs import CoreComponents, HyperParameters
from torchaid.core.protocols import SizedIterable
from torchaid.core.states import FitContext, BatchState


class LoggingCallback(CallbackInterface):
    def __init__(self):
        self.pbar = None

    def on_train_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        """エポック開始時に新しいプログレスバーを生成する"""
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(fit_context.train_dataloader),
            desc=f"Epoch {fit_context.global_state.current_epoch}",
            leave=True
        )

    def on_train_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Train] [Epoch {fit_context.train_dataloader}/{fit_context.hyper_parameters.max_epoch}]')

    def on_train_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_posfitx(self.train_display_items(core_components, fit_context, batch_state))

    def train_display_items(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> dict[str, str]:
        return {"loss": f"{batch_state.loss}"}

    def on_train_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def on_val_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        """エポック開始時に新しいプログレスバーを生成する"""
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(fit_context.val_dataloader),
            desc=f"Epoch {fit_context.global_state.current_epoch}",
            leave=True
        )

    def on_val_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Val] [Epoch {fit_context.train_dataloader}/{fit_context.hyper_parameters.max_epoch}]')

    def on_val_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_posfitx(self.val_display_items(core_components, fit_context, batch_state))

    def val_display_items(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> dict[str, str]:
        return {"loss": f"{batch_state.loss}"}

    def on_val_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def on_test_begin(self, core_components: CoreComponents, test_dataloader: SizedIterable[BatchState], hyper_parameters: HyperParameters) -> None:
        """エポック開始時に新しいプログレスバーを生成する"""
        # leave=True にすると、終わったバーが画面に残ります（学習履歴として見やすい）
        self.pbar = tqdm(
            total=len(test_dataloader),
            leave=True
        )

    def on_test_batch_start(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        """バッチ終了時にバーを1つ進め、Lossの数値を右側に表示する"""
        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_description(f'[Test]')

    def on_test_batch_end(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> None:
        if self.pbar is not None:
            self.pbar.set_posfitx(self.test_display_items(core_components, fit_context, batch_state))

    def test_display_items(self, core_components: CoreComponents, fit_context: FitContext, batch_state: BatchState) -> dict[str, str]:
        return {"loss": f"{batch_state.loss}"}

    def on_test_end(self, core_components: CoreComponents, test_dataloader: SizedIterable[BatchState], hyper_parameters: HyperParameters) -> None:
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
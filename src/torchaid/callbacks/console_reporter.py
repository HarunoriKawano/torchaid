from torchaid.core.callback import CallbackInterface
from torchaid.core.configs import CoreComponents
from torchaid.core.states import FitContext


class ConsoleReporter(CallbackInterface):
    @staticmethod
    def _strong_print(strings: list[str]):
        if not strings:
            return
        max_length = max([len(string) for string in strings])
        print(f"\n{'=' * (max_length + 4)}")
        for string in strings:
            print(f" {string:<{max_length}} ")
        print(f"{'=' * (max_length + 4)}\n")

    def on_fit_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:
        self._strong_print([
            "Training Start",
            f"Device:      {fit_context.hyper_parameters.device}",
            f"Remaining epochs: {fit_context.remaining_epoch}",
            f"Steps per epoch:  {len(fit_context.train_dataloader)}",
            f"Total steps:      {fit_context.remaining_epoch * len(fit_context.train_dataloader)}",
            f"Model parameters: {self._num_para:,}",
            f"Mixed precision:  {self._ls.mixed_precision} ({self._ls.precision_dtype})",
            f"Batch size:       {fit_context.hyper_parameters.batch_size}",
        ])
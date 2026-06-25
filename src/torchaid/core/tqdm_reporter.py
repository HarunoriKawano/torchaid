from tqdm import tqdm

from torchaid.core.callback import CallbackInterface
from torchaid.core.configs import CoreComponents
from torchaid.core.states import FitContext


class TQDMReporter(CallbackInterface):
    bar_format = '{n_fmt}/{total_fmt}: {percentage:3.0f}%, [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    def __init__(self, total_batches: int, current_epoch: int):
        self.total_batches = total_batches
        self.pbar = None

    def on_train_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:...

    def on_train_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:...

    def on_val_begin(self, core_components: CoreComponents, fit_context: FitContext) -> None:...

    def on_val_end(self, core_components: CoreComponents, fit_context: FitContext) -> None:...






from tqdm import tqdm


class TQDMReporter(ConsoleReporterInterface):
    def __init__(self, total_batches: int, current_epoch: int):
        self.total_batches = total_batches
        self.pbar = None

    def on_train_epoch(self, *args, **kwargs) -> None: pass

    def on_val_epoch(self, *args, **kwargs) -> None: pass

    def on_batch_start(self, ) -> None: pass

    def on_batch_end(self, *args, **kwargs) -> None: pass

    def on_fit_start(self, *args, **kwargs) -> None: pass

    def on_fit_end(self, *args, **kwargs) -> None: pass



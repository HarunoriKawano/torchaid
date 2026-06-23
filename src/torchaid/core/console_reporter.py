from abc import ABC, abstractmethod


class ConsoleReporterInterface:
    def on_fit_start(self, *args, **kwargs) -> None: pass

    @abstractmethod
    def on_batch_start(self, *args, **kwargs) -> None: pass

    @abstractmethod
    def on_epoch_start(self, *args, **kwargs) -> None: pass

    @abstractmethod
    def on_epoch_end(self, *args, **kwargs) -> None: pass

    @abstractmethod
    def on_batch_end(self, *args, **kwargs) -> None: pass

    @abstractmethod
    def on_fit_end(self, *args, **kwargs) -> None: pass

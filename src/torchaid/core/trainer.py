import os
import json
from pathlib import Path
from typing import Optional, Type
from itertools import islice

from pydantic import BaseModel
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from . import TaskModule, BaseSettings, BaseMetricCalculator, BaseMetrics, BaseInputs, BaseOutputs
from .configs import Mode

__all__ = ['TrainFramework']

class TrainFramework:
    """High-level training framework that manages the full model lifecycle.

    Handles the training loop, validation loop, mixed-precision support,
    gradient scaling, learning rate scheduling, checkpointing, metric logging,
    and early stopping.

    Attributes:
        bar_format (str): Format string passed to ``tqdm`` for progress bar display.
    """

    bar_format = '{n_fmt}/{total_fmt}: {percentage:3.0f}%, [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

    def __init__(
            self,
            model: TaskModule,
            ls: BaseSettings,
            metric_calculator: BaseMetricCalculator,
            optimizer: torch.optim.Optimizer,
            inputs_config: Type[BaseInputs],
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ):
        """Initializes the training framework and moves the model to the target device.

        Args:
            model (TaskModule): The task module to train.
            ls (BaseSettings): Training configuration (batch size, device, precision, etc.).
            metric_calculator (BaseMetricCalculator): Calculator responsible for
                accumulating and aggregating metrics each epoch.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            inputs_config (Type[BaseInputs]): The task's input schema class used
                to validate and cast raw dataloader batches via ``model_validate``.
            scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): Optional
                learning rate scheduler stepped after every training batch.
                Defaults to ``None``.

        Raises:
            TypeError: If any argument is not an instance of its expected type.
        """
        if not isinstance(model, TaskModule):
            raise TypeError(f"model must be a TaskModule instance, got {type(model).__name__}")
        if not isinstance(ls, BaseSettings):
            raise TypeError(f"ls must be a BaseSettings instance, got {type(ls).__name__}")
        if not isinstance(metric_calculator, BaseMetricCalculator):
            raise TypeError(f"metric_calculator must be a BaseMetricCalculator instance, got {type(metric_calculator).__name__}")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer must be a torch.optim.Optimizer instance, got {type(optimizer).__name__}")
        if not (isinstance(inputs_config, type) and issubclass(inputs_config, BaseInputs)):
            raise TypeError(f"inputs_config must be a subclass of BaseInputs, got {type(inputs_config).__name__}")
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(f"scheduler must be a LRScheduler instance or None, got {type(scheduler).__name__}")

        self._ls = ls
        self._device = torch.device(self._ls.device)
        self._task_module = model.to(self._device)

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._metric_calculator = metric_calculator
        self._inputs_config = inputs_config

        self._use_scaler = self._ls.mixed_precision and self._ls.precision_dtype == "float16"
        self._scaler = GradScaler(enabled=self._use_scaler)

        self._num_params: int = 0
        for p in self._task_module.parameters():
            self._num_params += p.numel()

    def train(self, train_dataset: Dataset, val_dataset: Dataset, save_dir: str):
        """Runs the full training loop with validation, checkpointing, and logging.

        The following steps are executed in order:

        1. **System check** — runs :meth:`_system_check` to verify the full
           pipeline with a small number of batches. Prints ``"Running system
           check..."`` before and ``"System check passed."`` after.
        2. **Checkpoint resume** — if ``checkpoint.pth`` already exists in
           ``save_dir``, :meth:`load_checkpoint` is called and the starting
           epoch is printed. Otherwise prints ``"Starting training from
           scratch."``.
        3. **Training summary** — prints a bordered block listing device,
           remaining epochs, steps per epoch, total steps, model parameter
           count, mixed-precision settings, and batch size.
        4. **Epoch loop** — for each remaining epoch, runs one training pass
           then one validation pass via :meth:`_loop`. After validation:

           - If :meth:`~torchaid.core.metrics.BaseMetricCalculator.check`
             returns ``True``, prints a best-model message and saves weights
             to ``best_checkpoint.pth`` via :meth:`save_checkpoint`.
           - Appends a row to ``epoch_log.csv``.
           - Prints a bordered metric summary for the epoch.
           - Saves a rolling checkpoint to ``checkpoint.pth`` and prints the
             save path.
           - If :meth:`~torchaid.core.metrics.BaseMetricCalculator.early_stopping`
             returns ``True``, prints an early-stopping message and exits the
             loop.

        5. **Completion** — prints a bordered ``"Training complete."`` block
           and writes ``results.json`` containing all final metrics plus
           ``num_params``.

        Args:
            train_dataset (Dataset): Dataset used for training.
            val_dataset (Dataset): Dataset used for validation.
            save_dir (str | os.PathLike): Directory where checkpoints, logs,
                and model weights are saved. Created automatically if it does
                not exist.

        Raises:
            TypeError: If ``train_dataset`` or ``val_dataset`` is not a
                ``Dataset``, or if ``save_dir`` is not a str or path-like.
            ValueError: If ``train_dataset`` or ``val_dataset`` is empty.
        """
        if not isinstance(train_dataset, Dataset):
            raise TypeError(f"train_dataset must be a Dataset instance, got {type(train_dataset).__name__}")
        if not isinstance(val_dataset, Dataset):
            raise TypeError(f"val_dataset must be a Dataset instance, got {type(val_dataset).__name__}")
        if not isinstance(save_dir, (str, os.PathLike)):
            raise TypeError(f"save_dir must be a str or path-like object, got {type(save_dir).__name__}")
        if len(train_dataset) == 0:
            raise ValueError("train_dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("val_dataset is empty")

        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = Path(os.path.join(save_dir, "checkpoint.pth"))

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self._ls.batch_size, num_workers=self._ls.cpu_num_works,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset, shuffle=False, batch_size=self._ls.batch_size, num_workers=self._ls.cpu_num_works,
            pin_memory=True
        )

        print("Running system check...")
        self._system_check(train_dataloader, val_dataloader, save_dir)
        print("System check passed.\n")

        if checkpoint_path.exists():
            self.load_checkpoint(str(checkpoint_path))
            print(f"Checkpoint found. Resuming from epoch {self._metric_calculator.metrics.epoch}/{self._ls.max_epoch_num}.\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")

        remaining_epochs = self._ls.max_epoch_num - self._metric_calculator.metrics.epoch
        self._strong_print([
            "Training Start",
            f"Device:      {self._ls.device}",
            f"Remaining epochs: {remaining_epochs}",
            f"Steps per epoch:  {len(train_dataloader)}",
            f"Total steps:      {remaining_epochs * len(train_dataloader)}",
            f"Model parameters: {self._num_params:,}",
            f"Mixed precision:  {self._ls.mixed_precision} ({self._ls.precision_dtype})",
            f"Batch size:       {self._ls.batch_size}",
        ])

        for _ in range(remaining_epochs):
            self._metric_calculator.reset()
            self._loop(train_dataloader, Mode.TRAIN)
            self._loop(val_dataloader, Mode.VAL)

            improved = self._metric_calculator.check()
            if improved:
                print(f"  New best model at epoch {self._metric_calculator.metrics.epoch}. Saving weights...")
                self.save_checkpoint(os.path.join(save_dir, "best_checkpoint.pth"))

            epoch_log_df = pd.DataFrame([self._metric_calculator.metrics.model_dump()])
            epoch_log_df.to_csv(
                os.path.join(save_dir, "epoch_log.csv"),
                encoding="utf-8",
                mode="w" if self._metric_calculator.metrics.epoch == 1 else "a",
                index=False,
                header=self._metric_calculator.metrics.epoch == 1
            )

            self._strong_print([f"{k}: {v}" for k, v in self._metric_calculator.metrics.model_dump().items()])

            self.save_checkpoint(str(checkpoint_path))
            print(f"  Checkpoint saved to '{checkpoint_path}'.")

            if self._metric_calculator.early_stopping():
                print(f"\nEarly stopping triggered at epoch {self._metric_calculator.metrics.epoch}.")
                break

        self._strong_print([
            "Training complete.",
            f"Results saved to '{save_dir}'.",
        ])

        result_dict = {
            **self._metric_calculator.metrics.model_dump(),
            "num_params": self._num_params
        }
        with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(result_dict, f)


    def test(self, test_dataset: Dataset) -> BaseMetrics:
        """Runs inference on the test dataset and returns aggregated metrics.

        Resets the metric calculator, prints ``"Starting test evaluation..."``,
        runs one pass over ``test_dataset`` via :meth:`_loop`, finalizes metrics
        with :meth:`~torchaid.core.metrics.BaseMetricCalculator.test`, and
        prints ``"Test evaluation complete."``.

        Args:
            test_dataset (Dataset): Dataset used for evaluation.

        Returns:
            BaseMetrics: The populated metrics object after the test loop.

        Raises:
            TypeError: If ``test_dataset`` is not a ``Dataset`` instance.
            ValueError: If ``test_dataset`` is empty.
        """
        if not isinstance(test_dataset, Dataset):
            raise TypeError(f"test_dataset must be a Dataset instance, got {type(test_dataset).__name__}")
        if len(test_dataset) == 0:
            raise ValueError("test_dataset is empty")

        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=self._ls.batch_size, num_workers=self._ls.cpu_num_works,
            pin_memory=True, persistent_workers=True
        )
        self._metric_calculator.reset()

        print("Starting test evaluation...")
        self._loop(test_dataloader, Mode.TEST)
        self._metric_calculator.test()
        print("Test evaluation complete.\n")
        return self._metric_calculator.metrics

    def _loop(self, dataloader, mode: Mode):
        """Iterates over a dataloader for one epoch in the given mode.

        Puts the model in training or evaluation mode, advances the epoch/step
        counters, and dispatches each batch to :meth:`_train_step` or
        :meth:`_eval_step` accordingly.

        Args:
            dataloader: An iterable of batches. Can be a ``DataLoader`` or a
                slice of one (e.g., from :func:`itertools.islice`).
            mode (Mode): Determines whether gradients are computed and whether
                the epoch counter is incremented.
        """
        if mode is Mode.TRAIN:
            self._metric_calculator.metrics.epoch += 1
            self._task_module.train()
        else:
            self._task_module.eval()
        with tqdm(dataloader, bar_format=self.bar_format) as pbar:
            if mode is Mode.TEST:
                pbar.set_description(f'[Test]')
            else:
                pbar.set_description(f'[{mode}] [Epoch {self._metric_calculator.metrics.epoch}/{self._ls.max_epoch_num}]')
            for data in pbar:
                if mode is Mode.TRAIN:
                    self._metric_calculator.metrics.step += 1
                    outputs, batch = self._train_step(data)
                    display_items = self._metric_calculator.train_step(outputs, batch)
                elif mode is Mode.VAL:
                    outputs, batch = self._eval_step(data, Mode.VAL)
                    display_items = self._metric_calculator.val_step(outputs, batch)
                else:
                    outputs, batch = self._eval_step(data, Mode.TEST)
                    display_items = self._metric_calculator.test_step(outputs, batch)
                pbar.set_postfix(display_items)


    def _train_step(self, batch: dict[str, torch.Tensor]):
        """Executes a single training step including forward pass and backpropagation.

        Validates the raw batch dict into the typed inputs schema, runs the
        forward pass under optional mixed-precision autocast, computes gradients,
        and updates the optimizer (and scaler when using ``float16``). Tensors are
        moved back to CPU after the step.

        Args:
            batch (dict[str, torch.Tensor]): Raw batch dictionary from the
                ``DataLoader``, keyed by field names of the inputs schema.

        Returns:
            tuple[BaseOutputs, BaseInputs]: The model outputs and the validated
                input batch, both on CPU.
        """
        batch = self._inputs_config.model_validate(batch)
        self._to_device(batch, torch.device(self._device))

        self._optimizer.zero_grad(set_to_none=True)
        dtype = torch.bfloat16 if self._ls.precision_dtype == "bfloat16" else torch.float16
        with autocast(device_type=self._device.type, enabled=self._ls.mixed_precision, dtype=dtype):
            outputs: BaseOutputs = self._task_module(Mode.TRAIN, batch=batch)

        if self._use_scaler:
            self._scaler.scale(outputs.loss).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            outputs.loss.backward()
            self._optimizer.step()

        if self._scheduler:
            self._scheduler.step()

        self._to_device(batch, torch.device("cpu"))
        self._to_device(outputs, torch.device("cpu"))

        return outputs, batch


    def _eval_step(self, batch: dict[str, torch.Tensor], mode: Mode):
        """Executes a single evaluation step without gradient computation.

        Validates the raw batch dict, runs the forward pass under optional
        mixed-precision autocast with ``torch.no_grad`` semantics, and moves
        tensors back to CPU.

        Args:
            batch (dict[str, torch.Tensor]): Raw batch dictionary from the
                ``DataLoader``.
            mode (Mode): Either ``Mode.VAL`` or ``Mode.TEST``. Passed directly
                to the model so it can adjust its output accordingly.

        Returns:
            tuple[BaseOutputs, BaseInputs]: The model outputs and the validated
                input batch, both on CPU.
        """
        batch = self._inputs_config.model_validate(batch)
        self._to_device(batch, torch.device(self._device))

        dtype = torch.bfloat16 if self._ls.precision_dtype == "bfloat16" else torch.float16
        with autocast(device_type=self._device.type, enabled=self._ls.mixed_precision, dtype=dtype):
            outputs = self._task_module(mode, batch=batch)

        self._to_device(batch, torch.device("cpu"))
        self._to_device(outputs, torch.device("cpu"))

        return outputs, batch


    @staticmethod
    def _strong_print(strings: list[str]):
        """Prints a bordered block of text to stdout for visual emphasis.

        Each item in ``strings`` is printed on its own line, padded to the
        length of the longest item and surrounded by ``=`` borders.

        Args:
            strings (list[str]): Lines of text to display inside the border.
        """
        if not strings:
            return
        max_length = max([len(string) for string in strings])
        print(f"\n{'=' * (max_length + 4)}")
        for string in strings:
            print(f" {string:<{max_length}} ")
        print(f"{'=' * (max_length + 4)}\n")

    @staticmethod
    def _to_device(values: BaseModel, device: torch.device):
        """Moves all ``torch.Tensor`` fields of a Pydantic model to the given device in-place.

        Args:
            values (BaseModel): A Pydantic model instance whose tensor fields
                should be transferred (e.g., a ``BaseInputs`` or ``BaseOutputs``
                subclass instance).
            device (torch.device): Target device (e.g., ``torch.device("cuda")``
                or ``torch.device("cpu")``).
        """
        for field_name, value in values:
            if isinstance(value, torch.Tensor):
                setattr(values, field_name, value.to(device))

    def load_model(self, path: str):
        """Loads model weights from a saved state-dict file.

        Prints ``"Loading model weights from '<path>'..."`` before loading and
        ``"Model loaded successfully."`` on completion.

        Args:
            path (str): File path to the ``.pth`` file containing only the
                model state dictionary (as saved by ``torch.save`` on
                ``model.state_dict()``). Use :meth:`load_checkpoint` to restore
                the full training state including optimizer and metrics.

        Raises:
            FileNotFoundError: If no file exists at ``path``.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"Loading model weights from '{path}'...")
        self._task_module.load_state_dict(torch.load(path, map_location=self._ls.device))
        print("Model loaded successfully.")

    def load_checkpoint(self, path: str):
        """Restores the full training state from a checkpoint file.

        Loads model weights, optimizer state, optional scheduler state, and
        metric values from a file previously written by :meth:`save_checkpoint`.
        Metric fields are applied via
        :meth:`~torchaid.core.metrics.BaseMetricCalculator.replace`, so only
        keys that match declared metric fields are restored; extra keys are
        silently ignored.

        Args:
            path (str): File path to the checkpoint ``.pth`` file.

        Raises:
            FileNotFoundError: If no file exists at ``path``.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path)
        self._task_module.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._metric_calculator.replace(checkpoint)
        if self._scheduler:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def save_checkpoint(self, path: str):
        """Saves the current training state to a checkpoint file.

        Writes model weights, optimizer state, optional scheduler state, and
        all metric fields to ``path`` using ``torch.save``. The resulting file
        can be restored later with :meth:`load_checkpoint`.

        Args:
            path (str): Destination file path for the checkpoint ``.pth`` file.
                Parent directories must already exist.
        """
        checkpoint = {
            'model_state_dict': self._task_module.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict() if self._scheduler else None,
            **self._metric_calculator.metrics.model_dump()
        }
        torch.save(checkpoint, path)

    def _system_check(self, train_dataloader: DataLoader, val_dataloader: DataLoader, save_dir: str, test_steps: int = 5):
        """Performs a sanity check by running a few steps before full training.

        Saves the current state to a temporary checkpoint
        ``<save_dir>/system_check.pth`` via :meth:`save_checkpoint`, resets the
        metric calculator, then runs ``test_steps`` training and ``test_steps``
        validation batches to verify that the model, optimizer, data pipeline,
        and metric calculator are all configured correctly. Prints progress via
        ``"Running N train steps and N val steps..."`` and displays a metric
        summary via :meth:`~torchaid.core.metrics.BaseMetricCalculator.system_check`.
        Afterwards, the initial state is fully restored from the temporary
        checkpoint via :meth:`load_checkpoint` so that epoch and step counters
        are reset to zero before actual training begins. The temporary file is
        deleted on completion.

        Args:
            train_dataloader (DataLoader): Training dataloader to sample from.
            val_dataloader (DataLoader): Validation dataloader to sample from.
            save_dir (str): Directory in which the temporary checkpoint is
                written. Must already exist.
            test_steps (int): Number of batches to run from each dataloader.
                Defaults to ``5``.
        """
        temp_path = Path(os.path.join(save_dir, "system_check.pth"))
        self.save_checkpoint(str(temp_path))

        self._metric_calculator.reset()

        print(f"  Running {test_steps} train steps and {test_steps} val steps...")
        limited_train_loader = islice(train_dataloader, test_steps)
        limited_val_loader = islice(val_dataloader, test_steps)

        self._loop(limited_train_loader, Mode.TRAIN)
        self._loop(limited_val_loader, Mode.VAL)
        self._metric_calculator.check()
        self._metric_calculator.system_check()

        self.load_checkpoint(str(temp_path))
        self._metric_calculator.reset()
        temp_path.unlink()

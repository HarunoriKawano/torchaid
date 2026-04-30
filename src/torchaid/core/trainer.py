import os
import json
from pathlib import Path
from typing import Optional
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
            inputs_config: BaseInputs,
            scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ):
        """Initializes the training framework and moves the model to the target device.

        Args:
            model (TaskModule): The task module to train.
            ls (BaseSettings): Training configuration (batch size, device, precision, etc.).
            metric_calculator (BaseMetricCalculator): Calculator responsible for
                accumulating and aggregating metrics each epoch.
            optimizer (torch.optim.Optimizer): Optimizer used for parameter updates.
            inputs_config (BaseInputs): An instance of the task's input schema used
                to validate and cast raw dataloader batches via ``model_validate``.
            scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): Optional
                learning rate scheduler stepped after every training batch.
                Defaults to ``None``.
        """
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

        Before training begins, a system sanity check is performed using a small
        number of batches. If a checkpoint exists in ``save_dir``, training resumes
        from that checkpoint. After each epoch the best model weights are saved,
        a CSV log is appended, and a checkpoint is written. Training terminates
        early if :meth:`~torchaid.core.metrics.BaseMetricCalculator.early_stopping`
        returns ``True``. A ``results.json`` file is written after the final epoch.

        Args:
            train_dataset (Dataset): Dataset used for training.
            val_dataset (Dataset): Dataset used for validation.
            save_dir (str): Directory path where checkpoints, logs, and model
                weights are saved. Created automatically if it does not exist.
        """
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

        self._system_check(train_dataloader, val_dataloader, checkpoint_path)

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self._task_module.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._metric_calculator.replace(checkpoint)
            if self._scheduler:
                self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._strong_print([
            "Training Start",
            f"Max epoch: {self._ls.max_epoch_num - self._metric_calculator.metrics.epoch}",
            f"Total step: {(self._ls.max_epoch_num - self._metric_calculator.metrics.epoch) * len(train_dataloader)}",
            f"Model size: {self._num_params}"
        ])

        for _ in range(self._ls.max_epoch_num - self._metric_calculator.metrics.epoch):
            self._metric_calculator.reset()
            self._loop(train_dataloader, Mode.TRAIN)
            self._loop(val_dataloader, Mode.VAL)

            if self._metric_calculator.check():
                torch.save(self._task_module.state_dict(), os.path.join(save_dir, "best_model.pth"))

            epoch_log_df = pd.DataFrame(self._metric_calculator.metrics.model_dump())
            epoch_log_df.to_csv(
                os.path.join(save_dir, "epoch_log.csv"),
                encoding="utf-8",
                mode="w" if self._metric_calculator.metrics.epoch == 1 else "a",
                index=False,
                header=self._metric_calculator.metrics.epoch == 1
            )

            self._strong_print(self._metric_calculator.metrics.model_dump())

            checkpoint = {
                'model_state_dict': self._task_module.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'scheduler_state_dict': self._scheduler.state_dict() if self._scheduler else None,
                **self._metric_calculator.metrics.model_dump()
            }
            torch.save(checkpoint, checkpoint_path)

            if self._metric_calculator.early_stopping():
                break

        result_dict = {
            **self._metric_calculator.metrics.model_dump(),
            "num_params": self._num_params
        }
        with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(result_dict, f)


    def test(self, test_dataset: Dataset) -> BaseMetrics:
        """Runs inference on the test dataset and returns aggregated metrics.

        The metric calculator is reset before the loop and
        :meth:`~torchaid.core.metrics.BaseMetricCalculator.test` is called
        afterwards to finalize the results.

        Args:
            test_dataset (Dataset): Dataset used for evaluation.

        Returns:
            BaseMetrics: The populated metrics object after the test loop.
        """
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=self._ls.batch_size, num_workers=self._ls.cpu_num_works,
            pin_memory=True, persistent_workers=True
        )
        self._metric_calculator.reset()

        self._loop(test_dataloader, Mode.TEST)
        self._metric_calculator.test()
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
                else:
                    outputs, batch = self._eval_step(data)
                    display_items = self._metric_calculator.val_step(outputs, batch)
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
            outputs: BaseOutputs = self._task_module(mode="train", batch = batch)

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


    def _eval_step(self, batch: dict[str, torch.Tensor]):
        """Executes a single evaluation step without gradient computation.

        Validates the raw batch dict, runs the forward pass under optional
        mixed-precision autocast with ``torch.no_grad`` semantics, and moves
        tensors back to CPU.

        Args:
            batch (dict[str, torch.Tensor]): Raw batch dictionary from the
                ``DataLoader``.

        Returns:
            tuple[BaseOutputs, BaseInputs]: The model outputs and the validated
                input batch, both on CPU.
        """
        batch = self._inputs_config.model_validate(batch)
        self._to_device(batch, torch.device(self._device))

        dtype = torch.bfloat16 if self._ls.precision_dtype == "bfloat16" else torch.float16
        with autocast(device_type=self._device.type, enabled=self._ls.mixed_precision, dtype=dtype):
            outputs = self._task_module(mode="eval", batch=batch)

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

        Args:
            path (str): File path to the ``.pth`` file containing the model
                state dictionary (as saved by ``torch.save``).
        """
        self._task_module.load_state_dict(torch.load(path, map_location=self._ls.device))

    def _system_check(self, train_dataloader: DataLoader, val_dataloader: DataLoader, checkpoint_path: Path, test_steps: int = 5):
        """Performs a sanity check by running a few steps before full training.

        Saves an initial checkpoint, then runs ``test_steps`` training and
        validation batches to verify that the model, optimizer, and data
        pipeline are all configured correctly. Prints current metrics at the end.

        Args:
            train_dataloader (DataLoader): Training dataloader to sample from.
            val_dataloader (DataLoader): Validation dataloader to sample from.
            checkpoint_path (Path): Path where the initial checkpoint is written.
            test_steps (int): Number of batches to process from each dataloader.
                Defaults to ``5``.
        """
        checkpoint = {
            'model_state_dict': self._task_module.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict() if self._scheduler else None,
            **self._metric_calculator.metrics.model_dump()
        }
        torch.save(checkpoint, checkpoint_path)

        limited_train_loader = islice(train_dataloader, test_steps)
        limited_val_loader = islice(val_dataloader, test_steps)

        self._loop(limited_train_loader, Mode.TRAIN)
        self._loop(limited_val_loader, Mode.VAL)
        self._metric_calculator.check()
        self._metric_calculator.system_check()

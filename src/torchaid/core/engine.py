import os
import shutil
import copy

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .states import FitContext, BatchState
from .callback import CallbackManager
from .configs import CoreComponents, HyperParameters
from .metrics import MetricInterface
from .data import ShortDataLoader


class Engine:
    core_components_file_name = "core_components.pth"
    global_state_file_name = "global_state.pth"
    system_check_iteration_num = 5

    def __init__(self, core_components: CoreComponents, callback_manager: CallbackManager, metrics: MetricInterface):
        self.core_components = core_components
        self.callback_manager = callback_manager
        self.metrics = metrics

    def fit(self, fit_context: FitContext, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.callback_manager.on_fit_begin(self.core_components, fit_context)

        for _ in range(fit_context.remaining_epoch):
            self._epoch_loop(fit_context, save_dir)

        self.callback_manager.on_fit_end(self.core_components, fit_context)

    def _epoch_loop(self, fit_context: FitContext, save_dir: str):
        # train step
        self.callback_manager.on_train_begin(self.core_components, fit_context)
        for batch_state in fit_context.train_dataloader:
            self._train_step(fit_context, batch_state)
        fit_context.one_epoch()
        self.callback_manager.on_train_end(self.core_components, fit_context)

        # val step
        self.callback_manager.on_val_begin(self.core_components, fit_context)
        self.metrics.reset()
        for batch_state in fit_context.val_dataloader:
            self._val_step(fit_context, batch_state)
        self.metrics.compute("val")

        # checkpoint save
        self.core_components.save(os.path.join(save_dir, self.core_components_file_name))
        fit_context.global_state.save(os.path.join(save_dir, self.global_state_file_name))

        # check result
        if fit_context.global_state.check_metric(self.metrics):
            self.core_components.task_module.save(save_dir)

        self.callback_manager.on_val_end(self.core_components, fit_context)


    def _train_step(self, fit_context: FitContext, batch_state: BatchState):
        self.callback_manager.on_train_batch_start(self.core_components, fit_context, batch_state)
        self.core_components.optimizer.zero_grad(set_to_none=True)
        batch_state.to(fit_context.hyper_parameters.device)

        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=fit_context.hyper_parameters.device.type, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.train_step(batch_state, fit_context)

        if batch_state.loss is None:
            raise TypeError("lossが更新されていません。")
        if fit_context.grad_scaler:
            fit_context.grad_scaler.scale(batch_state.loss).backward()
            fit_context.grad_scaler.step(self.core_components.optimizer)
            fit_context.grad_scaler.update()
        else:
            batch_state.loss.backward()
            self.core_components.optimizer.step()

        if fit_context.scheduler:
            fit_context.scheduler.step()

        batch_state.to(torch.device("cpu"))
        fit_context.one_step()
        self.callback_manager.on_train_batch_end(self.core_components, fit_context, batch_state)

    def _val_step(self, fit_context: FitContext, batch_state: BatchState):
        self.callback_manager.on_val_batch_start(self.core_components, fit_context, batch_state)
        batch_state.to(fit_context.hyper_parameters.device)

        dtype = getattr(torch, fit_context.hyper_parameters.amp) if fit_context.hyper_parameters.amp is not None else None
        with autocast(device_type=fit_context.hyper_parameters.device.type, enabled=bool(fit_context.hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.val_step(batch_state)

        batch_state.to(torch.device("cpu"))
        self.metrics.update(batch_state, "val")
        self.callback_manager.on_val_batch_end(self.core_components, fit_context, batch_state)

    def test(self, test_dataloader: DataLoader[BatchState], hyper_parameters: HyperParameters) -> MetricInterface:
        self.callback_manager.on_test_begin(self.core_components)
        self.metrics.reset()
        for batch_state in test_dataloader:
            self._test_step(hyper_parameters, batch_state)
        self.metrics.compute("test")
        return self.metrics

    def _test_step(self, hyper_parameters: HyperParameters, batch_state: BatchState):
        self.callback_manager.on_test_batch_start(self.core_components, batch_state)
        batch_state.to(hyper_parameters.device)

        dtype = getattr(torch, hyper_parameters.amp) if hyper_parameters.amp is not None else None
        with autocast(device_type=hyper_parameters.device.type, enabled=bool(hyper_parameters.amp), dtype=dtype):
            batch_state = self.core_components.task_module.val_step(batch_state)

        batch_state.to(torch.device("cpu"))
        self.metrics.update(batch_state, "test")
        self.callback_manager.on_test_batch_end(self.core_components, batch_state)

    def system_check(self, fit_context: FitContext, save_dir: str = "./system_check") -> None:
        copy_fit_context = copy.deepcopy(fit_context)
        copy_core_components = copy.deepcopy(self.core_components)
        os.makedirs(save_dir, exist_ok=True)

        copy_fit_context.train_dataloader = ShortDataLoader(copy_fit_context.train_dataloader, self.system_check_iteration_num)
        copy_fit_context.val_dataloader = ShortDataLoader(copy_fit_context.val_dataloader, self.system_check_iteration_num)

        self.callback_manager.on_fit_begin(self.core_components, copy_fit_context)
        self._epoch_loop(copy_fit_context, save_dir)
        self.callback_manager.on_fit_end(self.core_components, fit_context)

        shutil.rmtree(save_dir)
        self.core_components = copy_core_components

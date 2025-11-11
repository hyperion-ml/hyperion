"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.optim.swa_utils import SWALR, AveragedModel

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..optim import OptimizerFactory as OF
from ..torch_model import TorchModel
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF
from .torch_trainer_base import AMPDType, DDPType, TorchTrainerBase


class SingleModelTrainer(TorchTrainerBase):
    """Base Trainer class to train single neural network models

    Attributes:
        model: TorchModel object
        optim: Optimizer object or Dictionary of options to initialize the optimizer
        lrsched: Learning rate scheduler object or Dictionary of options to initialize the scheduler.
        wdsched: Weight decay scheduler object or Dictionary of options to initialize the scheduler.
        train_mode: str = "full",
        loss: optional loss class derived from nn.Module
        exp_path: experiment output path
        num_epochs: max. number of epochs
        cur_epoch: current epoch
        max_steps: max. number of steps
        cur_step:  current step
        grad_acc_steps: gradient accumulation steps to simulate larger batch size.
        eff_batch_size: effective batch size
        val_steps: steps between validation loops
        val_hours: max. number of hours between validation loops
        save_steps: steps between model saves
        save_hours: max. number of hours between model saves.
        device: gpu device
        loggers: LoggerList object, loggers write training progress to std. output and file.
        ddp: if True use distributed data parallel training
        ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
        cpu_offload: CPU offload of gradients when using fully sharded ddp
        use_amp: uses mixed precision training.
        amp_dtype: "float16" | "bfloat16"
        log_interval: number of optim. steps between log outputs
        use_tensorboard: use tensorboard logger
        use_wandb: use wandb logger
        wandb: wandb dictionary of options
        grad_clip: norm to clip gradients, if 0 there is no clipping
        grad_clip_norm: norm type to clip gradients
        swa_start: step to start doing SWA
        swa_lr: SWA learning rate
        swa_anneal_steps: SWA learning rate annealing steps
        swa_update_steps: steps between SWA model averagings
        bn_update_steps:  max. number of steps for updating BatchNorm after SWA
        input_key: Key of the batch_data returned by the data loader to use as model input
        target_key: Key of the batch_data returned by the data loader to use as model target
    """

    def __init__(
        self,
        model: TorchModel,
        optim: torch.optim.Optimizer,
        lrsched: Optional[LRS] = None,
        wdsched: Optional[WDS] = None,
        train_mode: str = "full",
        loss: Optional[nn.Module] = None,
        exp_path: PathLike = "./train",
        num_epochs: int = 100,
        cur_epoch: int = 0,
        max_steps: Optional[int] = None,
        cur_step: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        val_steps: Optional[int] = None,
        save_steps: Optional[int] = None,
        device: Union[torch.device, int, None] = None,
        loggers: Optional[LoggerList] = None,
        ddp: bool = False,
        ddp_type: DDPType = DDPType.DDP,
        cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        input_key="x",
        target_key="class_id",
    ):

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.model = model
        self.optim = optim
        self.lrsched = lrsched
        self.wdsched = wdsched
        self.train_mode = train_mode

        self.input_key = input_key
        self.target_key = target_key

        self.loss = loss
        if self.loss is not None:
            self.loss.to(self.device)

        self.set_train_mode()
        self.prepare_models_for_training()

    def prepare_models_for_training(self):
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
        ) = self._prepare_model_for_training(
            self.model,
            self.optim,
            self.lrsched,
            self.wdsched,
            self.device,
            self.use_amp,
            self.ddp,
            self.ddp_type,
            self.cpu_offload,
            self.do_swa,
            self.swa_lr,
            self.swa_anneal_steps,
        )
        self.grad_scaler = self.get_grad_scaler(self.use_amp, self.ddp, self.ddp_type)

    def set_train_mode(self):
        self.model.set_train_mode(self.train_mode)
        if self.rank == 0:
            logging.info(f"Model train mode: {self.model.train_mode}")
            logging.info(f"Parmeter summary for the model:")
            self.model.parameter_summary(verbose=True)
            logging.info(f"Parameter list for the model:")
            self.model.print_parameter_list()

    def on_epoch_begin(self):
        super().on_epoch_begin()

        if self.lr_scheduler is not None:
            # this is needed by cosine scheduler
            self.lr_scheduler.on_epoch_begin(
                self.cur_epoch, epoch_updates=self.save_steps
            )

        if self.wd_scheduler is not None:
            self.wd_scheduler.on_epoch_begin(self.cur_epoch)

    def on_epoch_end(self, logs):
        super().on_epoch_end(logs)
        if self.do_swa and self.cur_step >= self.swa_start:
            return

        if self.lr_scheduler is not None:
            self.lr_scheduler.on_epoch_end(logs)
        if self.wd_scheduler is not None:
            self.wd_scheduler.on_epoch_end()

    def on_swa_epoch_begin(self):
        super().on_swa_epoch_begin()
        self.model = self.swa_model.module

    def on_swa_epoch_end(self, logs):
        super().on_swa_epoch_end(logs)

    def on_train_loop_begin(self):
        self.model.train()

    def on_val_loop_begin(self):
        self.model.eval()

    def preprocess_data(self, batch_data):
        x_lengths_key = f"{self.input_key}_lengths"
        y_lengths_key = f"{self.target_key}_lengths"
        output_batch_data = {
            "id": batch_data["id"],
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        if x_lengths_key in batch_data:
            output_batch_data["audio_lengths"] = batch_data[x_lengths_key]
        if y_lengths_key in batch_data:
            output_batch_data["target_lengths"] = batch_data[y_lengths_key]
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(self, batch_data):
        output = self.model(**batch_data)
        loss = self.loss(output, batch_data["target"])
        return loss, output

    def compute_backward(self, loss):
        loss = loss.float()
        self.grad_scaler.scale(loss).backward()

    def zero_grad_optimizers(self):
        self.optimizer.zero_grad()

    def get_lrs(self):
        return self._get_lrs(self.optimizer)

    def get_wds(self):
        return self._get_wds(self.optimizer, self.wd_scheduler)

    def models_have_bn(self):
        return self.model.has_batchnorms()

    def update_models(self):
        """Updates the model and does gradding clipping."""
        if self.lr_scheduler is not None and not self.in_swa:
            self.lr_scheduler.on_opt_step()

        if self.wd_scheduler is not None:
            self.wd_scheduler.on_opt_step()

        grad_norm = self._update_model_by_optim(
            self.model,
            self.optimizer,
            self.grad_clip,
            self.grad_clip_norm,
            self.use_amp,
            self.grad_scaler,
        )
        self.grad_scaler.update()
        logs = {"grad_norm": grad_norm}
        return logs

    def update_swa_model(self):
        if (
            self.do_swa
            and self.cur_step >= self.swa_start
            and self.cur_step % self.swa_update_steps == 0
        ):
            self.in_swa = True
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

    def save_checkpoint(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
          partial: if True, it is saving in the middle of the epoch
        """
        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            # optimizer = cast(OSS, optimizer)
            self.optimizer.consolidate_state_dict()

        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
            logs=logs,
        )

        self.save_model_checkpoint_to_file("model", checkpoint)

    def save_swa_model(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
        """
        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
            logs=logs,
        )
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_model_ep%04d_%010d.pth" % (
            self.exp_path,
            self.cur_epoch,
            self.cur_step,
        )
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, epoch, step):
        checkpoint = self.load_model_checkpoint_from_file("model", epoch, step)
        logs = self._load_vars_from_checkpoint(checkpoint)
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.wd_scheduler,
            self.swa_model,
            self.swa_scheduler,
        )
        return logs

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(SingleModelTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_optim_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        OF.add_class_args(parser, prefix="optim")
        LRSF.add_class_args(parser, prefix="lrsched")
        WDSF.add_class_args(parser, prefix="wdsched")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_train_modes_args(parser, train_modes: List[str] = None):
        if train_modes is not None:
            parser.add_argument(
                "--train-mode",
                default="full",
                choices=train_modes,
                help=f"Available train modes for the model in {train_modes}",
            )

    @staticmethod
    def add_io_keys_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--input-key", default="audio_aug", help="dict. key for nnet input"
        )
        parser.add_argument(
            "--target-key", default="speaker", help="dict. key for nnet targets"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        SingleModelTrainer.add_optim_args(parser)
        SingleModelTrainer.add_io_keys_args(parser)
        SingleModelTrainer.add_train_modes_args(parser, train_modes=train_modes)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

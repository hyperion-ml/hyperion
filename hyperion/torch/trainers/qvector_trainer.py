"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict
from typing import Dict, List, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..metrics import CategoricalAccuracy
from ..models.qvectors import QVectorTrainMode
from ..narchs.hydra_heads import HydraClassifHeadOutput
from ..torch_model import TorchModel
from ..utils import MetricAcc, tensors_subset
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF
from .single_model_trainer import SingleModelTrainer
from .torch_trainer_base import AMPDType, DDPType

# from torch.distributed.elastic.multiprocessing.errors import record


class QVectorTrainer(SingleModelTrainer):
    """Trainer to train q-vector style models.

    Attributes:
        model: TorchModel object
        optim: Optimizer object or Dictionary of options to initialize the optimizer
        lrsched: Learning rate scheduler object or Dictionary of options to initialize the scheduler.
        wdsched: Weight decay scheduler object or Dictionary of options to initialize the scheduler.
        train_mode: str = "full",
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
        input_key="audio",
        target_key="speaker",
    ):

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.categorical_acc_metric = CategoricalAccuracy()

    # def preprocess_train_data(self, batch_data):
    #     # we get the keys of all augmented versions, except the non augmented one
    #     aug_keys = self.get_augs_keys(
    #         batch_data, self.input_key, skip=set(self.input_key)
    #     )
    #     x_lengths = batch_data[f"{self.input_key}_length"]
    #     y = batch_data[self.target_key]
    #     if aug_keys:
    #         # we concatenate all augmentations
    #         xs = []
    #         for key in aug_keys:
    #             xs.append(batch_data[key])
    #         x = torch.cat(xs, dim=0)
    #         x_lengths = torch.cat(len(aug_keys) * [x_lengths], dim=0)
    #         y = torch.cat(len(aug_keys) * [y], dim=0)
    #     if not aug_keys:
    #         x = batch_data[self.input_key]

    #     batch_data = {"x": x, "x_lengths": x_lengths, "y": y}
    #     batch_size = batch_data["x"].size(0)
    #     return batch_size, batch_data

    def preprocess_data(self, batch_data):
        x_lengths_key = f"{self.input_key}_lengths"
        # y_lengths_key = f"{self.target_key}_lengths"
        output_batch_data = {
            # "id": batch_data["id"],
            "audio": batch_data[self.input_key],
            "target": batch_data[self.target_key],
        }
        if x_lengths_key in batch_data:
            output_batch_data["audio_lengths"] = batch_data[x_lengths_key]
        # if y_lengths_key in batch_data:
        #     output_batch_data["target_lengths"] = batch_data[y_lengths_key]
        batch_size = output_batch_data["audio"].size(0)
        return batch_size, output_batch_data

    def compute_forward(self, batch_data):
        batch_output = self.model(**batch_data)
        loss = batch_output.head_output.loss
        return loss, batch_output

    def compute_metrics(self, batch_output, batch_data):
        batch_metrics = ODict()
        if isinstance(batch_output.head_output, HydraClassifHeadOutput):
            categorical_acc = self.categorical_acc_metric(
                batch_output.head_output.logits, batch_data["target"]
            )

            batch_metrics["categorical_acc"] = categorical_acc.item()
        else:
            logging.warning(
                "QVectorTrainer: compute_metrics: Unknown head_output type %s"
                % type(batch_output.head_output)
            )

        return batch_metrics

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(QVectorTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        SingleModelTrainer.add_optim_args(parser)
        SingleModelTrainer.add_io_keys_args(parser)
        train_modes = QVectorTrainMode.choices()
        SingleModelTrainer.add_train_modes_args(parser, train_modes=train_modes)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

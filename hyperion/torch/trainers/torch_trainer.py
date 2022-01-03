"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import contextlib
from collections import OrderedDict as ODict
from enum import Enum
from jsonargparse import ArgumentParser, ActionParser
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.distributed as dist

from fairscale.optim.grad_scaler import ShardedGradScaler

from ..utils import MetricAcc, TorchDDP, FairShardedDDP, FairFullyShardedDDP
from ..loggers import LoggerList, CSVLogger, ProgLogger, TensorBoardLogger, WAndBLogger
from ..optim import OptimizerFactory as OF
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..lr_schedulers import LRScheduler as LRS


class DDPType(str, Enum):
    DDP = "ddp"
    OSS_DDP = "oss_ddp"
    OSS_SHARDED_DDP = "oss_sharded_ddp"
    FULLY_SHARDED_DDP = "fully_sharded_ddp"


ddp_choices = [o.value for o in DDPType]


class TorchTrainer(object):
    """Base Trainer class to train basic neural network models

    Attributes:
      model: model object.
      loss: nn.Module loss class
      optim: pytorch optimizer object or optimizer options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object
      loggers: LoggerList object, loggers write training progress to std. output and file.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      train_mode: training mode in ['train', 'ft-full', 'ft-last-layer']
      use_amp: uses mixed precision training.
      log_interval: number of optim. steps between log outputs
      use_tensorboard: use tensorboard logger
      use_wandb: use wandb logger
      wandb: wandb dictionary of options
      grad_clip: norm to clip gradients, if 0 there is no clipping
      grad_clip_norm: norm type to clip gradients
      swa_start: epoch to start doing swa
      swa_lr: SWA learning rate
      swa_anneal_epochs: SWA learning rate anneal epochs
      cpu_offload: CPU offload of gradients when using fully sharded ddp
    """

    def __init__(
        self,
        model,
        loss,
        optim={},
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        device=None,
        metrics=None,
        lrsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        train_mode="train",
        use_amp=False,
        log_interval=10,
        use_tensorboard=False,
        use_wandb=False,
        wandb={},
        grad_clip=0,
        grad_clip_norm=2,
        swa_start=0,
        swa_lr=1e-3,
        swa_anneal_epochs=10,
        cpu_offload=False,
    ):

        self.model = model
        # self.optimizer = optim
        self.loss = loss
        self.epochs = epochs
        self.cur_epoch = cur_epoch
        self.grad_acc_steps = grad_acc_steps
        self.exp_path = Path(exp_path)

        if loggers is None:
            self.loggers = self._default_loggers(
                log_interval, use_tensorboard, use_wandb, wandb
            )
        elif isinstance(loggers, list):
            self.loggers = LoggerList(loggers)
        else:
            self.loggers = loggers

        # self.lr_scheduler = lr_scheduler

        self.metrics = metrics
        self.device = device
        self.train_mode = train_mode
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.grad_clip_norm = grad_clip_norm
        self.swa_start = swa_start
        self.do_swa = swa_start > 0
        self.swa_lr = swa_lr
        self.swa_anneal_epochs = swa_anneal_epochs
        self.amp_args = {}

        if device is not None:
            self.model.to(device)
            if loss is not None:
                self.loss.to(device)

        self.ddp = ddp
        self.ddp_type = ddp_type
        self.rank = 0
        self.world_size = 1
        if ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            if ddp_type == DDPType.DDP or ddp_type == DDPType.OSS_DDP:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with distributed-data-parallel"
                    )
                oss = False if ddp_type == DDPType.DDP else True
                self.optimizer = self._make_optimizer(optim, self.model, oss=oss)
                self.model = TorchDDP(
                    self.model, device_ids=[device], output_device=device
                )
            elif ddp_type == DDPType.OSS_SHARDED_DDP:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with fair sharded-distributed-data-parallel"
                    )
                self.optimizer = self._make_optimizer(optim, self.model, oss=True)
                self.model = FairShardedDDP(self.model, self.optimizer)
            else:
                if self.rank == 0:
                    logging.info(
                        "training in multiple gpus with fair fully-sharded-distributed-data-parallel"
                    )
                # syncbathcnorm is not supported here, it raises exception
                self.model = FairFullyShardedDDP(
                    self.model,
                    mixed_precision=self.use_amp,
                    move_params_to_cpu=cpu_offload,
                )
                self.optimizer = self._make_optimizer(optim, self.model, oss=False)

        else:
            self.optimizer = self._make_optimizer(optim, self.model)

        # make the learning rate scheduler
        self.lr_scheduler = self._make_lr_sched(lrsched, self.optimizer)

        if self.use_amp:
            if ddp and ddp_type != DDPType.DDP:
                if self.rank == 0:
                    logging.info(
                        "using automatic mixed precision training with sharded-grad-scaler"
                    )
                self.grad_scaler = ShardedGradScaler()
            else:
                if self.rank == 0:
                    logging.info(
                        "using automatic mixed precision training with grad-scaler"
                    )
                self.grad_scaler = amp.GradScaler()
            self.amp_autocast = amp.autocast
        else:
            self.amp_autocast = contextlib.nullcontext

        self.in_swa = False
        if self.do_swa:
            if self.rank == 0:
                logging.info("init SWA model")
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(
                self.optimizer, swa_lr=self.swa_lr, anneal_epochs=self.swa_anneal_epochs
            )

    def fit(self, train_data, val_data=None):
        """Training function, it performs the training and validation epochs

        Args:
          train_data: PyTorch data loader for the training loop
          val_data: PyTorch data loader for the validation loop
        """
        self.exp_path.mkdir(parents=True, exist_ok=True)
        # if not os.path.exists(self.exp_path):
        #     os.makedirs(self.exp_path)

        if self.do_swa and self.cur_epoch >= self.swa_start:
            self.in_swa = True

        val_logs = {}
        self.loggers.on_train_begin(epochs=self.epochs)
        for epoch in range(self.cur_epoch, self.epochs):

            self.loggers.on_epoch_begin(epoch, batches=len(train_data))
            if self.lr_scheduler is not None:
                # this is needed by cosine scheduler
                epoch_updates = int(len(train_data) / self.grad_acc_steps)
                self.lr_scheduler.on_epoch_begin(epoch, epoch_updates=epoch_updates)

            logs = self.train_epoch(train_data)
            if val_data is not None:
                val_logs = self.validation_epoch(val_data)
                logs.update(val_logs)

            self.cur_epoch += 1

            self.loggers.on_epoch_end(logs)
            if self.do_swa and self.cur_epoch >= self.swa_start:
                self.in_swa = True
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                if self.lr_scheduler is not None:
                    self.lr_scheduler.on_epoch_end(logs)

            self.save_checkpoint(logs)

        if self.in_swa:
            self.loggers.on_epoch_begin(self.cur_epoch, batches=len(train_data))
            self.model = self.swa_model.module
            logs = self.bn_update_epoch(train_data)

            if val_data is not None:
                val_logs = self.validation_epoch(val_data)
                logs.update(val_logs)

            self.cur_epoch += 1
            self.loggers.on_epoch_end(logs)
            self.save_swa_model(logs)

    def set_train_mode(self):
        if self.train_mode == "train":
            self.model.train()
        else:
            self.model.train_mode(self.train_mode)

    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.set_train_mode()
        for batch, (data, target) in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)
            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]
            with self.amp_autocast():
                output = self.model(data)
                loss = self.loss(output, target).mean() / self.grad_acc_steps

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.lr_scheduler is not None and not self.in_swa:
                    self.lr_scheduler.on_opt_step()
                self.update_model()

            self._reduce_metric(loss)
            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            for k, metric in self.metrics.items():
                batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            logs["lr"] = self._get_lr()
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)
            # total_batches += 1

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        logs["lr"] = self._get_lr()
        return logs

    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs
        """

        metric_acc = MetricAcc(self.device)
        batch_metrics = ODict()
        with torch.no_grad():
            if swa_update_bn:
                log_tag = "train_"
                self.set_train_mode()
            else:
                log_tag = "val_"
                self.model.eval()

            for batch, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]

                with self.amp_autocast():
                    output = self.model(data, **self.amp_args)
                    loss = self.loss(output, target)

                batch_metrics["loss"] = loss.mean().item()
                for k, metric in self.metrics.items():
                    batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    def bn_update_epoch(self, data_loader):
        logs = self.validation_epoch(data_loader, swa_update_bn=True)
        logs["lr"] = self._get_lr()
        return logs

    def _clip_grad_norm(self, model, optim, grad_clip, grad_clip_norm):
        if self.ddp:
            if self.ddp_type == DDPType.DDP:
                nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, norm_type=grad_clip_norm
                )
                return
            if self.ddp_type == DDPType.FULLY_SHARDED_DDP:
                # we have to use the member function in FullyShardedDDP class
                model.clip_grad_norm_(grad_clip, norm_type=grad_clip_norm)
                return
            else:
                # not sure about this but it looks like
                # we have to use the member function in the OSS optimizer wrapper
                optim.clip_grad_norm(grad_clip, norm_type=grad_clip_norm)

        # if no DDP clip normally
        nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, norm_type=grad_clip_norm
        )

    def update_model(self):

        if self.use_amp:
            if self.grad_clip > 0:
                self.grad_scaler.unscale_(self.optimizer)
                self._clip_grad_norm(
                    self.model, self.optimizer, self.grad_clip, self.grad_clip_norm
                )

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            if self.grad_clip > 0:
                self._clip_grad_norm(
                    self.model, self.optimizer, self.grad_clip, self.grad_clip_norm
                )

            self.optimizer.step()

    def _make_optimizer(self, optim, model, oss=False):
        if isinstance(optim, torch.optim.Optimizer):
            return optim

        assert isinstance(optim, dict)
        opt_args = OF.filter_args(**optim)
        opt_args["oss"] = oss
        if self.rank == 0:
            logging.info("optimizer args={}".format(opt_args))
        optimizer = OF.create(model.parameters(), **opt_args)
        return optimizer

    def _make_lr_sched(self, lr_sched, optim):
        if lr_sched is None or isinstance(lr_sched, LRS):
            return lr_sched

        assert isinstance(lr_sched, dict)
        args = LRSF.filter_args(**lr_sched)
        if self.rank == 0:
            logging.info("lr scheduler args={}".format(args))
        lr_sched = LRSF.create(optim, **args)
        return lr_sched

    def _default_loggers(self, log_interval, use_tensorboard, use_wandb, wandb):
        """Creates the default data loaders"""
        prog_log = ProgLogger(interval=log_interval)
        csv_log = CSVLogger(self.exp_path / "train.log", append=True)
        loggers = [prog_log, csv_log]
        if use_tensorboard:
            loggers.append(
                TensorBoardLogger(self.exp_path / "tb", interval=log_interval)
            )
        if use_wandb:
            loggers.append(
                WAndBLogger(
                    **wandb, path=self.exp_path / "wandb", interval=log_interval
                )
            )
        return LoggerList(loggers)

    def _get_lr(self):
        """Returns the current learning rate to show in the loggers"""
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def checkpoint(self, logs=None):
        """Creates a checkpoint of the training, to save and posterior recovery

        Args:
          logs: logs containing the current value of the metrics.
        """
        checkpoint = {
            "epoch": self.cur_epoch,
            "rng_state": torch.get_rng_state(),
            "model_cfg": self.model.get_config(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_state_dict": self.loss.state_dict()
            if self.loss is not None
            else None,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if logs is not None:
            checkpoint["logs"] = logs

        if self.in_swa:
            checkpoint["swa_model_state_dict"] = self.swa_model.state_dict()
            checkpoint["swa_scheduler_state_dict"] = self.swa_scheduler.state_dict()

        return checkpoint

    def save_checkpoint(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
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
        checkpoint = self.checkpoint(logs)
        file_path = "%s/model_ep%04d.pth" % (self.exp_path, self.cur_epoch)

        torch.save(checkpoint, file_path)

    def save_swa_model(self, logs=None):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
        """
        if self.rank != 0:
            return

        checkpoint = self.checkpoint(logs)
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_model_ep%04d.pth" % (self.exp_path, self.cur_epoch)

        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path):
        """Loads a training checkpoint from file.

        Args:
           file_path: checkpoint file path
        """
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        rng_state = checkpoint["rng_state"]
        torch.set_rng_state(rng_state)
        if self.rank > 0:
            # this will make sure that each process produces different data
            # when using ddp
            dummy = torch.rand(1000 * self.rank)
            del dummy

        self.cur_epoch = checkpoint["epoch"]
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.loss is not None:
            self.loss.load_state_dict(checkpoint["loss_state_dict"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # if self.use_amp:
        #    amp.load_state_dict(checkpoint['amp'])
        if self.do_swa:
            if "swa_model_state_dict" in checkpoint:
                self.swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
                self.swa_scheduler.load_state_dict(
                    checkpoint["swa_scheduler_state_dict"]
                )
            else:
                self.swa_scheduler = SWALR(
                    self.optimizer,
                    swa_lr=self.swa_lr,
                    anneal_epochs=self.swa_anneal_epochs,
                )

        logs = None
        if "logs" in checkpoint:
            logs = checkpoint["logs"]

        del checkpoint
        if self.device is not None:
            torch.cuda.empty_cache()

        return logs

    def load_last_checkpoint(self):
        """Loads the last training checkpoint in the experiment dir."""
        for epoch in range(self.epochs, 0, -1):
            file_path = "%s/model_ep%04d.pth" % (self.exp_path, epoch)
            if os.path.isfile(file_path):
                return self.load_checkpoint(file_path)

        return None

    @staticmethod
    def filter_args(**kwargs):
        valid_args = (
            "grad_acc_steps",
            "epochs",
            "log_interval",
            "use_amp",
            "ddp_type",
            "grad_clip",
            "swa_start",
            "swa_lr",
            "swa_anneal_epochs",
            "exp_path",
            "optim",
            "lrsched",
            "cpu_offload",
            "use_tensorboard",
            "use_wandb",
            "wandb",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=[]):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "optim" not in skip:
            OF.add_class_args(parser, prefix="optim")

        if "lrsched" not in skip:
            LRSF.add_class_args(parser, prefix="lrsched")

        parser.add_argument(
            "--grad-acc-steps",
            type=int,
            default=1,
            help="gradient accumulation batches before weigth update",
        )
        parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
        parser.add_argument(
            "--log-interval",
            type=int,
            default=10,
            help="how many batches to wait before logging training status",
        )
        parser.add_argument(
            "--use-tensorboard",
            action="store_true",
            default=False,
            help="use tensorboard logger",
        )
        parser.add_argument(
            "--use-wandb", action="store_true", default=False, help="use wandb logger"
        )
        parser.add_argument("--wandb.project", default=None, help="wandb project name")
        parser.add_argument("--wandb.group", default=None, help="wandb group name")
        parser.add_argument("--wandb.name", default=None, help="wandb display name")
        # parser.add_argument(
        #     '--wandb.path', default=None,
        #     help='wandb directory')
        parser.add_argument(
            "--wandb.mode",
            default="online",
            choices=["online", "offline"],
            help="wandb mode (online, offline)",
        )

        parser.add_argument(
            "--ddp-type",
            default="ddp",
            choices=ddp_choices,
            help="DDP type in {}".format(ddp_choices),
        )
        parser.add_argument(
            "--use-amp",
            action="store_true",
            default=False,
            help="use mixed precision training",
        )
        parser.add_argument(
            "--cpu-offload",
            action="store_true",
            default=False,
            help="CPU offload of gradients when using fully_sharded_ddp",
        )
        parser.add_argument(
            "--grad-clip", type=float, default=0, help="gradient clipping norm value"
        )
        parser.add_argument(
            "--grad-clip-norm",
            default=2,
            choices=["inf", 1, 2],
            help="gradient clipping norm type",
        )
        parser.add_argument(
            "--swa-start",
            type=int,
            default=0,
            help="start epoch for SWA, if 0 it does not use SWA",
        )
        parser.add_argument(
            "--swa-lr", type=float, default=1e-3, help="learning rate for SWA phase"
        )
        parser.add_argument(
            "--swa-anneal-epochs",
            type=int,
            default=10,
            help="SWA learning rate anneal epochs",
        )

        parser.add_argument("--exp-path", help="experiment path")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='trainer options')

    add_argparse_args = add_class_args

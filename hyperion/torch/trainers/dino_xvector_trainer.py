"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import os
from collections import OrderedDict as ODict

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

from ...utils.misc import filter_func_args
from ..optim import ExpMovingAvg as EMA
from ..utils import MetricAcc, TorchDDP, tensors_subset
from .torch_trainer import DDPType, TorchTrainer


class DINOXVectorTrainer(TorchTrainer):
    """Trainer to train x-vector style models.

    Attributes:
      model: x-Vector model object.
      optim: pytorch optimizer object or options dict
      epochs: max. number of epochs
      exp_path: experiment output path
      cur_epoch: current epoch
      grad_acc_steps: gradient accumulation steps to simulate larger batch size.
      device: cpu/gpu device
      metrics: extra metrics to compute besides cxe.
      lrsched: learning rate scheduler object or options dict
      teacher_optim: teacher EMA momentum
      loggers: LoggerList object, loggers write training progress to std. output and file.
               If None, it uses default loggers.
      ddp: if True use distributed data parallel training
      ddp_type: type of distributed data parallel in  (ddp, oss_ddp, oss_shared_ddp)
      loss: if None, it uses cross-entropy
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
      save_interval_steps: number of steps between model saves, if None only saves at the end of the epoch
      cpu_offload: CPU offload of gradients when using fully sharded ddp
      input_key: dict. key for nnet input.
      target_key: dict. key for nnet targets.
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        loss,
        optim,
        teacher_optim,
        epochs=100,
        exp_path="./train",
        cur_epoch=0,
        grad_acc_steps=1,
        eff_batch_size=None,
        device=None,
        metrics=None,
        lrsched=None,
        wdsched=None,
        loggers=None,
        ddp=False,
        ddp_type="ddp",
        train_mode="full",
        freeze_output_layer_steps=3000,
        use_amp=False,
        log_interval=1000,
        use_tensorboard=False,
        use_wandb=False,
        wandb={},
        grad_clip=0,
        grad_clip_norm=2,
        swa_start=0,
        swa_lr=1e-3,
        swa_anneal_epochs=10,
        save_interval_steps=None,
        cpu_offload=False,
        input_key="x",
    ):
        super_args = filter_func_args(super().__init__, locals())
        self.teacher_model = teacher_model
        self.teacher_optim = teacher_optim
        self.freeze_output_layer_steps = freeze_output_layer_steps
        super().__init__(student_model, **super_args)

    def prepare_models_for_training(self):
        super().prepare_models_for_training()
        self.teacher_model, self.teacher_optimizer = self._prepare_model_for_ema(
            self.teacher_model,
            self.teacher_optim,
            self.device,
            self.ddp,
        )

    def _prepare_model_for_ema(self, model, optim, device, ddp):
        if device is not None:
            model.to(device)

        optimizer = EMA(model.parameters(), **optim)

        if ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model, optimizer

    def set_train_mode(self):
        super().set_train_mode()
        self.teacher_model.freeze()

    @torch.no_grad()
    def update_teacher_model(self):
        self.teacher_optimizer.step(self.model.parameters())

    @staticmethod
    def get_augs_keys(batch, base_key, subset, skip=set()):
        base_key = f"{base_key}_{subset}"
        keys = []

        chunk_idx = 0
        while True:
            found_chunk = 0
            chunk_key = f"{base_key}_{chunk_idx}"
            if chunk_key in batch:
                if chunk_key not in skip:
                    keys.append(chunk_key)
                found_chunk = True
            aug_idx = 0
            while True:
                aug_key = f"{chunk_key}_aug_{aug_idx}"
                if aug_key in batch:
                    if aug_key not in skip:
                        keys.append(aug_key)

                    aug_idx += 1
                    found_chunk = True
                else:
                    break

            if not found_chunk:
                break

            chunk_idx += 1

        return keys

    @record
    def train_epoch(self, data_loader):
        """Training epoch loop

        Args:
          data_loader: pytorch data loader returning features and class labels.
        """
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()
        self.teacher_model.train()
        self.loss.update_temp(self.cur_epoch)
        self.loss.train()

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            teacher_keys = self.get_augs_keys(data, self.input_key, "teacher")
            student_keys = self.get_augs_keys(data, self.input_key, "student")
            with amp.autocast(enabled=self.use_amp):
                with torch.no_grad():
                    teacher_data = tensors_subset(data, teacher_keys, self.device)
                    batch_size = teacher_data[0].size(0)
                    num_teacher_crops = len(teacher_data)
                    teacher_data = torch.cat(teacher_data, dim=0)
                    teacher_out = self.teacher_model(teacher_data)

                if num_teacher_crops > 1:
                    student_out1 = self.model(teacher_data)

                student_data = tensors_subset(data, student_keys, self.device)
                num_student_crops = len(student_data)
                student_data = torch.cat(student_data, dim=0)
                student_out2 = self.model(student_data)
                assert not torch.any(torch.isnan(teacher_out)), "teacher is nan"
                assert not torch.any(torch.isinf(teacher_out)), "teacher is inf"
                assert not torch.any(torch.isnan(student_out1)), "s1 is nan"
                assert not torch.any(torch.isinf(student_out1)), "s1 is inf"
                assert not torch.any(torch.isnan(student_out2)), "s2 is nan"
                assert not torch.any(torch.isinf(student_out2)), "s2 is inf"
                if num_teacher_crops > 1:
                    student_out = torch.cat((student_out1, student_out2), dim=0)
                    num_student_crops += num_teacher_crops
                else:
                    student_out = student_out2

                loss = (
                    self.loss(
                        student_out, teacher_out, num_student_crops, num_teacher_crops
                    )
                    / self.grad_acc_steps
                )
                assert not torch.isnan(
                    loss
                ), f"loss is nan {batch} {torch.mean(teacher_out)} {torch.mean(student_out1)} {torch.mean(student_out2)}"

            if self.use_amp:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                self.cur_batch = batch + 1
                if self.freeze_output_layer_steps > self.global_step:
                    self.model.cancel_output_layer_grads()

                self.update_model()
                self.update_teacher_model()
                self.save_checkpoint(partial=True)

            batch_metrics["loss"] = loss.item() * self.grad_acc_steps
            # for k, metric in self.metrics.items():
            #     batch_metrics[k] = metric(output, target)

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            lrs = self._get_lrs()
            logs.update(lrs)
            logs["ema_momentum"] = self.teacher_optimizer.momentum
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        lrs = self._get_lrs()
        logs.update(lrs)
        logs.update(self._get_wds())
        logs["ema_momentum"] = self.teacher_optimizer.momentum
        return logs

    @torch.no_grad()
    def validation_epoch(self, data_loader, swa_update_bn=False):
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader return input/output pairs.
          sw_update_bn: wheter or not, update batch-norm layers in SWA.
        """
        metric_acc = MetricAcc(self.device)
        batch_metrics = ODict()
        self.teacher_model.eval()
        self.loss.eval()

        if swa_update_bn:
            log_tag = "train_"
            self.model.train()
        else:
            log_tag = "val_"
            self.model.eval()

        for batch, data in enumerate(data_loader):
            teacher_keys = self.get_augs_keys(data, self.input_key, "teacher")
            student_keys = self.get_augs_keys(data, self.input_key, "student")
            with amp.autocast(enabled=self.use_amp):
                teacher_data = tensors_subset(data, teacher_keys, self.device)
                batch_size = teacher_data[0].size(0)
                num_teacher_crops = len(teacher_data)
                teacher_data = torch.cat(teacher_data, dim=0)
                teacher_out = self.teacher_model(teacher_data)

                if num_teacher_crops > 1:
                    student_out1 = self.model(teacher_data)

                student_data = tensors_subset(data, student_keys, self.device)
                num_student_crops = len(student_data)
                student_data = torch.cat(student_data, dim=0)
                student_out2 = self.model(student_data)
                if num_teacher_crops > 1:
                    student_out = torch.cat((student_out1, student_out2), dim=0)
                    num_student_crops += num_teacher_crops
                else:
                    student_out = student_out2

                loss = self.loss(
                    student_out, teacher_out, num_student_crops, num_teacher_crops
                )

                batch_metrics["loss"] = loss.item()
                # for k, metric in self.metrics.items():
                #     batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    def _old_load_checkpoint(self, checkpoint):
        self.teacher_model.load_state_dict(checkpoint["teacher_model_state_dict"])
        # self.teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
        self.teacher_optimizer.load_state_dict(
            checkpoint["teacher_optimizer_state_dict"]
        )
        return super()._load_checkpoint(checkpoint)

    def _load_checkpoint(self, checkpoint, teacher_checkpoint):
        self.teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
        self.teacher_optimizer.load_state_dict(
            teacher_checkpoint["optimizer_state_dict"]
        )
        return super()._load_checkpoint(checkpoint)

    def load_checkpoint(self, epoch, step):
        checkpoint = self.load_model_checkpoint("model", epoch, step)
        teacher_checkpoint = self.load_model_checkpoint("teacher_model", epoch, step)
        return self._load_checkpoint(checkpoint, teacher_checkpoint)

    def checkpoint(self, logs=None):
        checkpoint = super().checkpoint(logs)
        self.teacher_model.train()
        checkpoint["teacher_model_state_dict"] = self.teacher_model.state_dict()
        checkpoint["teacher_optimizer_state_dict"] = self.teacher_optimizer.state_dict()
        return checkpoint

    def teacher_checkpoint(self, logs=None):
        """Creates a checkpoint of the teacher model, to save and posterior recovery

        Args:
          logs: logs containing the current value of the metrics.
        """
        self.teacher_model.train()
        checkpoint = {
            "epoch": self.cur_epoch,
            "batch": self.cur_batch,
            "global_step": self.global_step,
            "model_cfg": self.teacher_model.get_config(),
            "model_state_dict": self.teacher_model.state_dict(),
            "optimizer_state_dict": self.teacher_optimizer.state_dict(),
        }

        if logs is not None:
            checkpoint["logs"] = logs

        return checkpoint

    def save_checkpoint(self, logs=None, partial: bool = False):
        """Saves a checkpoint of the training status

        Args:
          logs: logs containing the current value of the metrics.
          partial: if True, it is saving in the middle of the epoch
        """
        if partial and not self.save_partial_checkpoint():
            return

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
        self.save_model_checkpoint("model", checkpoint, partial=partial)

        teacher_checkpoint = self.teacher_checkpoint(logs)
        self.save_model_checkpoint("teacher_model", teacher_checkpoint, partial=partial)

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(DINOXVectorTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, train_modes=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip.add("teacher_key")
        TorchTrainer.add_class_args(parser, train_modes=train_modes)
        EMA.add_class_args(parser, prefix="teacher_optim")
        parser.add_argument(
            "--freeze-output-layer-steps",
            default=1500,
            type=int,
            help="freeze the output layer during the first updates of the model",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

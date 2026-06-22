"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import os
from collections import OrderedDict as ODict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.distributed.elastic.multiprocessing.errors import record

from ...utils.misc import filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import Logger, LoggerList
from ..optim import ExpMovingAvg as EMA
from ..utils import MetricAcc, tensors_subset
from .legacy_torch_trainer import AMPDType, LegacyTorchTrainer


class DINOXVectorTrainer(LegacyTorchTrainer):
    """Trainer to train x-vector style models.

    Attributes:
      model: Student model being trained.
      teacher_model: Teacher model updated by EMA.
      loss: DINO loss module.
      cosine_loss: Optional auxiliary cosine loss module.
      optim: Student optimizer configuration.
      teacher_optim: Teacher EMA configuration.
      epochs: Maximum number of epochs.
      exp_path: Experiment output path.
      cur_epoch: Current epoch.
      grad_acc_steps: Gradient accumulation steps.
      eff_batch_size: Desired effective batch size.
      device: Training device.
      metrics: Extra metrics to compute besides the loss.
      lrsched: Learning-rate scheduler configuration.
      wdsched: Weight-decay scheduler configuration.
      loggers: None, a list of loggers, or a LoggerList instance.
      ddp: Whether distributed training is enabled.
      ddp_type: Distributed backend selector.
      train_mode: Training mode.
      freeze_output_layer_steps: Number of steps to keep the output layer frozen.
      freeze_teacher: Whether the teacher is frozen.
      use_amp: Whether mixed precision training is enabled.
      amp_dtype: AMP dtype.
      log_interval: Interval between log writes.
      use_tensorboard: Whether TensorBoard logging is enabled.
      use_wandb: Whether W&B logging is enabled.
      wandb: W&B configuration.
      grad_clip: Gradient clipping threshold.
      grad_clip_norm: Gradient clipping norm type.
      swa_start: Epoch at which SWA starts.
      swa_lr: SWA learning rate.
      swa_anneal_epochs: SWA annealing epochs.
      save_interval_steps: Step interval for partial checkpoints.
      input_key: Input key for dict batches.
    """

    def __init__(
        self,
        student_model: HyperTorchModel,
        teacher_model: HyperTorchModel,
        loss: nn.Module,
        optim: Dict[str, Any],
        teacher_optim: Dict[str, Any],
        cosine_loss: Optional[nn.Module] = None,
        epochs: int = 100,
        exp_path: str = "./train",
        cur_epoch: int = 0,
        grad_acc_steps: int = 1,
        eff_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        metrics: Optional[Dict[str, Any]] = None,
        lrsched: Optional[Dict[str, Any]] = None,
        wdsched: Optional[Dict[str, Any]] = None,
        loggers: Optional[List[Logger] | LoggerList] = None,
        ddp: bool = False,
        ddp_type: str = "ddp",
        train_mode: str = "full",
        freeze_output_layer_steps: int = 3000,
        freeze_teacher: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        log_interval: int = 1000,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, Any] = {},
        grad_clip: float = 0,
        grad_clip_norm: float = 2,
        swa_start: int = 0,
        swa_lr: float = 1e-3,
        swa_anneal_epochs: int = 10,
        save_interval_steps: Optional[int] = None,
        input_key: str = "x",
    ) -> None:
        """Initializes a DINO x-vector trainer.

        Args:
          student_model: Student model to train.
          teacher_model: Teacher model updated by EMA.
          loss: DINO loss module.
          optim: Student optimizer configuration.
          teacher_optim: Teacher EMA configuration.
          cosine_loss: Optional auxiliary cosine loss module.
          epochs: Number of training epochs.
          exp_path: Output directory for checkpoints and logs.
          cur_epoch: Starting epoch.
          grad_acc_steps: Gradient accumulation factor.
          eff_batch_size: Desired effective batch size.
          device: Training device.
          metrics: Additional metric callables.
          lrsched: Learning-rate scheduler or configuration.
          wdsched: Weight-decay scheduler or configuration.
          loggers: None, a list of loggers, or a LoggerList instance.
          ddp: Enables distributed training.
          ddp_type: Distributed backend selector.
          train_mode: Model train mode.
          freeze_output_layer_steps: Steps to keep the output layer frozen.
          freeze_teacher: Whether the teacher model is frozen.
          use_amp: Enables automatic mixed precision.
          amp_dtype: AMP dtype name.
          log_interval: Batch interval between log writes.
          use_tensorboard: Enables TensorBoard logging.
          use_wandb: Enables W&B logging.
          wandb: Weights & Biases options.
          grad_clip: Gradient clip value.
          grad_clip_norm: Gradient clip norm type.
          swa_start: Epoch at which SWA starts.
          swa_lr: SWA learning rate.
          swa_anneal_epochs: SWA annealing epochs.
          save_interval_steps: Partial checkpoint interval.
          input_key: Input key for dict batches.
        """
        super_args = filter_func_args(super().__init__, locals())
        self.teacher_model = teacher_model
        self.teacher_optim = teacher_optim
        self.freeze_output_layer_steps = freeze_output_layer_steps
        self.freeze_teacher = freeze_teacher
        self.cosine_loss = cosine_loss
        super().__init__(student_model, **super_args)

    def prepare_models_for_training(self) -> None:
        """Moves the student and teacher models to the training device."""
        super().prepare_models_for_training()
        self.teacher_model, self.teacher_optimizer = self._prepare_model_for_ema(
            self.teacher_model,
            self.teacher_optim,
            self.device,
            self.ddp,
            self.freeze_teacher,
        )

    def _prepare_model_for_ema(
        self,
        model: HyperTorchModel,
        optim: Dict[str, Any],
        device: Optional[torch.device],
        ddp: bool,
        frozen: bool,
    ) -> Tuple[HyperTorchModel, Optional[EMA]]:
        """Moves the teacher to device and builds its EMA updater.

        Args:
          model: Teacher model.
          optim: EMA configuration dictionary.
          device: Target torch device.
          ddp: Whether DDP is enabled.
          frozen: Whether the teacher should remain frozen.

        Returns:
          The moved teacher model and its EMA optimizer, if any.
        """
        if device is not None:
            model.to(device)

        if frozen:
            return model, None

        optimizer = EMA(model.parameters(), **optim)
        if ddp:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        return model, optimizer

    def set_train_mode(self) -> None:
        """Sets train mode for the student and freezes the teacher."""
        super().set_train_mode()
        self.teacher_model.freeze()

    @torch.no_grad()
    def update_teacher_model(self) -> None:
        """Applies one EMA update to the teacher model."""
        if not self.freeze_teacher:
            self.teacher_optimizer.step(self.model.parameters())

    @staticmethod
    def get_augs_keys(
        batch: Dict[str, Any], base_key: str, subset: str, skip: Optional[set] = None
    ) -> List[str]:
        """Collects augmentation keys for a given subset.

        Args:
          batch: Batch dictionary.
          base_key: Base input key.
          subset: Subset suffix, for example ``teacher`` or ``student``.
          skip: Optional set of keys to exclude.

        Returns:
          List of batch keys matching the requested augmentation family.
        """
        skip = skip or set()
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
    def train_epoch(self, data_loader: Any) -> Dict[str, Any]:
        """Training epoch loop

        Args:
          data_loader: PyTorch data loader returning augmented views.

        Returns:
          Dictionary with training metrics.
        """
        metric_acc = MetricAcc(device=self.device)
        batch_metrics = ODict()
        self.model.train()
        if self.freeze_teacher:
            self.teacher_model.eval()
        else:
            self.teacher_model.train()
        self.loss.update_temp(self.cur_epoch)
        self.loss.train()
        if self.cosine_loss is not None:
            self.cosine_loss.update_scale(self.cur_epoch)

        for batch, data in enumerate(data_loader):
            self.loggers.on_batch_begin(batch)

            if batch % self.grad_acc_steps == 0:
                self.optimizer.zero_grad()

            teacher_keys = self.get_augs_keys(data, self.input_key, "teacher")
            student_keys = self.get_augs_keys(data, self.input_key, "student")
            with amp.autocast(
                enabled=self.use_amp,
                dtype=self.amp_dtype,
                device_type=self.device.type,
            ):
                with torch.no_grad():
                    teacher_data = tensors_subset(data, teacher_keys, self.device)
                    batch_size = teacher_data[0].size(0)
                    num_teacher_crops = len(teacher_data)
                    teacher_data = torch.cat(teacher_data, dim=0)
                    teacher_out = self.teacher_model(teacher_data)
                    if torch.any(torch.isnan(teacher_out.logits)):
                        logging.warning(f"teacher logits are nan")
                    # assert not torch.any(
                    #     torch.isnan(teacher_out.logits)
                    # ), "teacher is nan"
                    # assert not torch.any(
                    #     torch.isinf(teacher_out.logits)
                    # ), "teacher is inf"

                if num_teacher_crops > 1:
                    student_out1 = self.model(teacher_data)
                    if torch.any(torch.isnan(student_out1.logits)):
                        logging.warning(f"student-1 logits are nan")
                    # assert not torch.any(torch.isnan(student_out1.logits)), "s1 is nan"
                    # assert not torch.any(torch.isinf(student_out1.logits)), "s1 is inf"

                student_data = tensors_subset(data, student_keys, self.device)
                num_student_crops = len(student_data)
                student_data = torch.cat(student_data, dim=0)
                student_out2 = self.model(student_data)
                if torch.any(torch.isnan(student_out2.logits)):
                    logging.warning(f"student-2 logits are nan")
                # assert not torch.any(torch.isnan(student_out2.logits)), "s2 is nan"
                # assert not torch.any(torch.isinf(student_out2.logits)), "s2 is inf"
                if num_teacher_crops > 1:
                    student_out_logits = torch.cat(
                        (student_out1.logits, student_out2.logits), dim=0
                    )
                    if self.cosine_loss is not None:
                        student_out_embeds = torch.cat(
                            (student_out1.xvector, student_out2.xvector), dim=0
                        )
                    num_student_crops += num_teacher_crops
                else:
                    student_out_logits = student_out2.logits
                    student_out_embeds = student_out2.xvector

                loss_dino = self.loss(
                    student_out_logits,
                    teacher_out.logits.detach(),
                    num_student_crops,
                    num_teacher_crops,
                )
                loss = loss_dino
                if self.cosine_loss is not None:
                    scaled_loss_cosine, loss_cosine = self.cosine_loss(
                        student_out_embeds,
                        teacher_out.xvector.detach(),
                        num_student_crops,
                        num_teacher_crops,
                    )
                    loss = loss_dino + scaled_loss_cosine

                loss = loss / self.grad_acc_steps
                # assert not torch.isnan(
                #     loss
                # ), f"loss is nan {batch} {torch.mean(teacher_out)} {torch.mean(student_out1)} {torch.mean(student_out2)}"

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
            if self.cosine_loss is not None:
                batch_metrics["loss_dino"] = loss_dino.item()
                batch_metrics["loss_cosine"] = loss_cosine.item()

            metric_acc.update(batch_metrics, batch_size)
            logs = metric_acc.metrics
            lrs = self._get_lrs()
            logs.update(lrs)

            if self.teacher_optimizer is not None:
                logs["ema_momentum"] = self.teacher_optimizer.momentum
            self.loggers.on_batch_end(logs=logs, batch_size=batch_size)

        logs = metric_acc.metrics
        logs = ODict(("train_" + k, v) for k, v in logs.items())
        lrs = self._get_lrs()
        logs.update(lrs)
        logs.update(self._get_wds())
        if self.teacher_optimizer is not None:
            logs["ema_momentum"] = self.teacher_optimizer.momentum
        if self.grad_scaler is not None:
            logs["grad_scale"] = self.grad_scaler._scale.item()
        return logs

    @torch.no_grad()
    def validation_epoch(
        self, data_loader: Any, swa_update_bn: bool = False
    ) -> Dict[str, Any]:
        """Validation epoch loop

        Args:
          data_loader: PyTorch data loader returning augmented views.
          swa_update_bn: Whether to update batch-norm layers for SWA.

        Returns:
          Dictionary with validation metrics.
        """
        metric_acc = MetricAcc(self.device)
        batch_metrics = ODict()
        self.teacher_model.eval()
        self.loss.eval()

        log_tag = "train_" if swa_update_bn else "val_"
        if swa_update_bn:
            self.model.train()
        else:
            self.model.eval()

        for batch, data in enumerate(data_loader):
            teacher_keys = self.get_augs_keys(data, self.input_key, "teacher")
            student_keys = self.get_augs_keys(data, self.input_key, "student")
            with amp.autocast(
                enabled=self.use_amp,
                dtype=self.amp_dtype,
                device_type=self.device.type,
            ):
                teacher_data = tensors_subset(data, teacher_keys, self.device)
                batch_size = teacher_data[0].size(0)
                num_teacher_crops = len(teacher_data)
                teacher_data = torch.cat(teacher_data, dim=0)
                teacher_out = self.teacher_model(teacher_data)
                # assert not torch.any(torch.isnan(teacher_out.logits)), "teacher is nan"
                # assert not torch.any(torch.isinf(teacher_out.logits)), "teacher is inf"

                if num_teacher_crops > 1:
                    student_out1 = self.model(teacher_data)
                    # assert not torch.any(torch.isnan(student_out1.logits)), "s1 is nan"
                    # assert not torch.any(torch.isinf(student_out1.logits)), "s1 is inf"

                student_data = tensors_subset(data, student_keys, self.device)
                num_student_crops = len(student_data)
                student_data = torch.cat(student_data, dim=0)
                student_out2 = self.model(student_data)
                # assert not torch.any(torch.isnan(student_out2.logits)), "s2 is nan"
                # assert not torch.any(torch.isinf(student_out2.logits)), "s2 is inf"
                if num_teacher_crops > 1:
                    student_out_logits = torch.cat(
                        (student_out1.logits, student_out2.logits), dim=0
                    )
                    if self.cosine_loss is not None:
                        student_out_embeds = torch.cat(
                            (student_out1.xvector, student_out2.xvector), dim=0
                        )
                    num_student_crops += num_teacher_crops
                else:
                    student_out_logits = student_out2.logits
                    student_out_embeds = student_out2.xvector

                loss_dino = self.loss(
                    student_out_logits,
                    teacher_out.logits,
                    num_student_crops,
                    num_teacher_crops,
                )
                loss = loss_dino
                if self.cosine_loss is not None:
                    scaled_loss_cosine, loss_cosine = self.cosine_loss(
                        student_out_embeds,
                        teacher_out.xvector,
                        num_student_crops,
                        num_teacher_crops,
                    )
                    loss = loss_dino + scaled_loss_cosine

                batch_metrics["loss"] = loss.item()
                if self.cosine_loss is not None:
                    batch_metrics["loss_dino"] = loss_dino.item()
                    batch_metrics["loss_cosine"] = loss_cosine.item()
                # for k, metric in self.metrics.items():
                #     batch_metrics[k] = metric(output, target)

                metric_acc.update(batch_metrics, batch_size)

        logs = metric_acc.metrics
        logs = ODict((log_tag + k, v) for k, v in logs.items())
        return logs

    def _old_load_checkpoint(
        self, checkpoint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Loads an older checkpoint format.

        Args:
          checkpoint: Serialized checkpoint payload.

        Returns:
          Saved logs if present, otherwise ``None``.
        """
        self.teacher_model.load_state_dict(checkpoint["teacher_model_state_dict"])
        # self.teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
        self.teacher_optimizer.load_state_dict(
            checkpoint["teacher_optimizer_state_dict"]
        )
        return super()._load_checkpoint(checkpoint)

    def _load_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        teacher_checkpoint: Optional[Dict[str, Any]],
        loss_checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Loads student, teacher, and DINO loss checkpoints.

        Args:
          checkpoint: Student checkpoint payload.
          teacher_checkpoint: Teacher checkpoint payload.
          loss_checkpoint: DINO loss checkpoint payload.

        Returns:
          Saved logs if present, otherwise ``None``.
        """
        if teacher_checkpoint is not None:
            self.teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
            self.teacher_optimizer.load_state_dict(
                teacher_checkpoint["optimizer_state_dict"]
            )
        if loss_checkpoint is not None:
            self.loss.load_state_dict(loss_checkpoint["model_state_dict"])
        return super()._load_checkpoint(checkpoint)

    def load_checkpoint(self, epoch: int, step: int) -> Optional[Dict[str, Any]]:
        """Loads checkpoints for the student, teacher, and DINO loss.

        Args:
          epoch: Checkpoint epoch index.
          step: Checkpoint step index.

        Returns:
          Saved logs if present, otherwise ``None``.
        """
        checkpoint = self.load_model_checkpoint("model", epoch, step)
        if not self.freeze_teacher:
            teacher_checkpoint = self.load_model_checkpoint(
                "teacher_model", epoch, step
            )
        else:
            teacher_checkpoint = None
        try:
            loss_checkpoint = self.load_model_checkpoint("dino_loss", epoch, step)
        except:
            logging.warning(
                "dino loss checkpoint not found, initial center will be zero-vector"
            )
            loss_checkpoint = None
        return self._load_checkpoint(checkpoint, teacher_checkpoint, loss_checkpoint)

    # def checkpoint(self, logs=None):
    #     checkpoint = super().checkpoint(logs)
    #     # self.teacher_model.train()
    #     # checkpoint["teacher_model_state_dict"] = self.teacher_model.state_dict()
    #     # checkpoint["teacher_optimizer_state_dict"] = self.teacher_optimizer.state_dict()
    #     return checkpoint

    def teacher_checkpoint(
        self, logs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates a checkpoint of the teacher model, to save and posterior recovery

        Args:
          logs: Logs containing the current value of the metrics.

        Returns:
          Serializable teacher checkpoint dictionary.
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

    def dino_loss_checkpoint(
        self, logs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates a checkpoint for the DINO loss state.

        Args:
          logs: Logs containing the current value of the metrics.

        Returns:
          Serializable DINO loss checkpoint dictionary.
        """
        self.loss.train()
        checkpoint = {
            "epoch": self.cur_epoch,
            "batch": self.cur_batch,
            "global_step": self.global_step,
            "model_state_dict": self.loss.state_dict(),
        }
        return checkpoint

    def save_checkpoint(
        self, logs: Optional[Dict[str, Any]] = None, partial: bool = False
    ) -> None:
        """Saves a checkpoint of the training status

        Args:
          logs: Logs containing the current value of the metrics.
          partial: If ``True``, saves in the middle of the epoch.
        """
        if partial and not self.save_partial_checkpoint():
            return

        if self.rank != 0:
            return

        checkpoint = self.checkpoint(logs)
        self.save_model_checkpoint("model", checkpoint, partial=partial)

        if not self.freeze_teacher:
            teacher_checkpoint = self.teacher_checkpoint(logs)
            self.save_model_checkpoint(
                "teacher_model", teacher_checkpoint, partial=partial
            )

        loss_checkpoint = self.dino_loss_checkpoint()
        self.save_model_checkpoint("dino_loss", loss_checkpoint, partial=partial)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments accepted by this trainer.

        Returns:
          Dictionary of filtered arguments.
        """
        args = filter_func_args(DINOXVectorTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: Any,
        prefix: Optional[str] = None,
        train_modes: Optional[List[str]] = None,
        skip: Optional[set] = None,
    ) -> None:
        """Adds command-line arguments for the DINO trainer.

        Args:
          parser: Destination argument parser.
          prefix: Optional namespace prefix.
          train_modes: Allowed train-mode values.
          skip: Optional set of argument groups to skip.
        """
        skip = skip or set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip.add("teacher_key")
        LegacyTorchTrainer.add_class_args(parser, train_modes=train_modes, skip=skip)
        EMA.add_class_args(parser, prefix="teacher_optim")
        parser.add_argument(
            "--freeze-output-layer-steps",
            default=1500,
            type=int,
            help="freeze the output layer during the first updates of the model",
        )
        parser.add_argument(
            "--freeze-teacher",
            default=False,
            action=ActionYesNo,
            help="use a pre-trained frozen teacher",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

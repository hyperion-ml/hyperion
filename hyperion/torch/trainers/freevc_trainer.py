"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import glob
import logging
import math
import os
import re
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.amp as amp
import torch.distributed as dist
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..losses import (
    AudioDiscriminatorAdvLoss,
    AudioGeneratorAdvLoss,
    FeatureMatchingLoss,
)
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..models.audio_discrimitator.audio_multi_discriminator import (
    AudioDiscriminatorTrainMode,
)
from ..models.freevc.freevc import FreeVCFwdMode, FreeVCTrainMode
from ..narchs.audio_feats_mvn import AudioFeatsMVN
from ..optim import OptimizerFactory as OF
from ..torch_model import TorchModel
from ..utils.misc import rand_slice_audio_segments, slice_segments
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF
from .torch_trainer_base import AMPDType, DDPType, TorchTrainerBase


class FreeVCTrainer(TorchTrainerBase):
    """Trainer for FreeVC voice conversion models.

    This trainer handles the training of both the voice conversion (generator) and
    discriminator models used in adversarial training setups. It supports mixed precision,
    SWA, DDP, and advanced logging through TensorBoard and W&B.

    Attributes:
        vc_model: Generator model for voice conversion.
        discrim_model: Discriminator model for adversarial training.
        xvector_model: Pretrained speaker embedding model.
        audio_feats: Feature extractor for audio (e.g., log-mel spectrograms).
        vc_optim: Optimizer for the generator.
        discrim_optim: Optimizer for the discriminator.
        vc_lrsched: Learning rate scheduler for the generator.
        discrim_lrsched: Learning rate scheduler for the discriminator.
        vc_wdsched: Weight decay scheduler for the generator.
        discrim_wdsched: Weight decay scheduler for the discriminator.
        vc_train_mode: Training mode for the generator.
        discrim_train_mode: Training mode for the discriminator.
        exp_path (Path): Path to save training logs, checkpoints, and logs.
        num_epochs (int): Total number of training epochs.
        cur_epoch (int): Current training epoch.
        max_steps (int, optional): Maximum number of training steps (overrides num_epochs if set).
        cur_step (int): Current training step (incremented each optimizer update).
        grad_acc_steps (int): Number of batches to accumulate gradients over before optimizer step.
        eff_batch_size (int, optional): Effective total batch size across GPUs and grad accumulation.
        val_steps (int, optional): Number of steps between validation evaluations.
        val_hours (float, optional): Number of hours between validation evaluations (alternative to val_steps).
        save_steps (int, optional): Number of steps between model checkpoint saves.
        save_hours (float, optional): Number of hours between checkpoint saves (alternative to save_steps).
        device (torch.device or int, optional): Device to run training on (e.g. 'cuda:0').
        loggers (LoggerList): Logger interface for training output (e.g., console, file, TensorBoard).
        ddp (bool): Whether to use Distributed Data Parallel (DDP) training.
        ddp_type (DDPType): Type of DDP implementation to use (standard, OSS, Sharded, FullySharded).
        cpu_offload (bool): Whether to offload parameters/gradients to CPU in FullyShardedDDP.
        use_amp (bool): Enables mixed-precision training using AMP (Automatic Mixed Precision).
        amp_dtype (AMPDType): Data type for AMP (float16 or bfloat16).
        log_interval (int): Number of steps between log output (for loggers).
        log_gpu_usage (bool): Whether to log GPU usage (memory, utilization) during training.
        use_tensorboard (bool): Enables TensorBoard logger.
        use_wandb (bool): Enables Weights & Biases logger.
        wandb (dict): Configuration for W&B logging (project, name, etc.).
        grad_clip (float): Maximum gradient norm to clip (0 disables clipping).
        grad_clip_norm (str or int): Norm type for gradient clipping (e.g., 1, 2, or 'inf').
        swa_start (int): Step at which to begin Stochastic Weight Averaging (SWA).
        swa_lr (float): Learning rate to use during SWA phase.
        swa_anneal_steps (int): Number of steps over which to anneal SWA LR.
        swa_update_steps (int): Number of steps between averaging model weights in SWA.
        bn_update_steps (int): Max number of batches to use for batchnorm statistics update after SWA.
        input_audio_key: Batch key for source audio.
        target_audio_key: Batch key for target audio.
        loss_mel_weight: Weight for the mel spectrogram loss.
        loss_kl_weight: Weight for the KL divergence loss.
        loss_gen_adv_weight: Weight for adversarial generator loss.
        loss_fm_weight: Weight for feature matching loss.
        gen_segment_duration: Duration in seconds for VC generation segments.
        num_val_log_samples: Max number of samples to log during validation.
    """

    def __init__(
        self,
        vc_model: TorchModel,
        discrim_model: TorchModel,
        xvector_model: TorchModel,
        audio_feats: AudioFeatsMVN,
        vc_optim: torch.optim.Optimizer,
        discrim_optim: torch.optim.Optimizer,
        vc_lrsched: Optional[LRS] = None,
        discrim_lrsched: Optional[LRS] = None,
        vc_wdsched: Optional[WDS] = None,
        discrim_wdsched: Optional[WDS] = None,
        vc_train_mode: FreeVCTrainMode = FreeVCTrainMode.HF_FEATS_FROZEN_NOGRAD,
        discrim_train_mode: str = AudioDiscriminatorTrainMode.FULL,
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
        log_gpu_usage: bool = False,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        discrim_grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        input_audio_key="audio",
        target_audio_key="audio",
        loss_mel_weight: float = 20.0,
        loss_kl_weight: float = 1.0,
        loss_gen_adv_weight: float = 1.0,
        loss_fm_weight: float = 1.0,
        gen_segment_duration: float = 0.64,
        num_val_log_samples: int = 10,
    ):

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        if isinstance(audio_feats, dict):
            audio_feats = AudioFeatsMVN(**audio_feats)

        if self.rank == 0:
            logging.info("audio_feats={}".format(audio_feats))

        self.vc_model = vc_model
        self.discrim_model = discrim_model
        self.xvector_model = xvector_model
        self.audio_feats = audio_feats
        self.vc_optim = vc_optim
        self.vc_lrsched = vc_lrsched
        self.vc_wdsched = vc_wdsched
        self.discrim_optim = discrim_optim
        self.discrim_lrsched = discrim_lrsched
        self.discrim_wdsched = discrim_wdsched
        self.discrim_grad_clip = discrim_grad_clip
        self.vc_train_mode = vc_train_mode
        self.discrim_train_mode = discrim_train_mode
        self.input_audio_key = input_audio_key
        self.target_audio_key = target_audio_key
        self.loss_mel_weight = loss_mel_weight
        self.loss_kl_weight = loss_kl_weight
        self.loss_gen_adv_weight = loss_gen_adv_weight
        self.loss_fm_weight = loss_fm_weight
        self.gen_segment_duration = gen_segment_duration
        self.num_val_log_samples = num_val_log_samples
        self.cur_val_log_samples = 0

        self.set_train_mode()
        self.audio_feats.to(self.device)
        self.xvector_model.to(self.device)
        self.prepare_models_for_training()
        self.l1_loss = nn.L1Loss()
        self.discrim_adv_loss = AudioDiscriminatorAdvLoss()
        self.gen_adv_loss = AudioGeneratorAdvLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.ckpt_search_name = "vc_model"

    def prepare_models_for_training(self):
        """Initializes optimizers, schedulers, and SWA for both VC and discriminator models.

        Uses the `_prepare_model_for_training` helper for both models and sets up
        the `grad_scaler` for mixed precision.
        """
        (
            self.vc_model,
            self.vc_optimizer,
            self.vc_lr_scheduler,
            self.vc_wd_scheduler,
            self.swa_vc_model,
            self.swa_vc_scheduler,
        ) = self._prepare_model_for_training(
            self.vc_model,
            self.vc_optim,
            self.vc_lrsched,
            self.vc_wdsched,
            self.device,
            self.use_amp,
            self.ddp,
            self.ddp_type,
            self.cpu_offload,
            self.do_swa,
            self.swa_lr,
            self.swa_anneal_steps,
        )
        (
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_lr_scheduler,
            self.discrim_wd_scheduler,
            _,
            _,
        ) = self._prepare_model_for_training(
            self.discrim_model,
            self.discrim_optim,
            self.discrim_lrsched,
            self.discrim_wdsched,
            self.device,
            self.use_amp,
            self.ddp,
            self.ddp_type,
            self.cpu_offload,
            False,
        )
        self.grad_scaler = self.get_grad_scaler()

    def set_train_mode(self):
        """Applies the selected training modes to the generator and discriminator.

        Also logs parameter summaries and parameter lists if running on rank 0.
        """
        self.vc_model.set_train_mode(self.vc_train_mode)
        self.discrim_model.set_train_mode(self.discrim_train_mode)
        if self.rank == 0:
            logging.info(f"VC model train mode: {self.vc_train_mode}")
            logging.info(f"Parameter summary for VC model:")
            self.vc_model.parameter_summary(verbose=True)
            logging.info(f"VC model parameter list:")
            self.vc_model.print_parameter_list()
            logging.info(f"Discrim model train mode: {self.discrim_train_mode}")
            logging.info(f"Parameter summary for Discrim model:")
            self.discrim_model.parameter_summary(verbose=True)
            logging.info(f"Discrim model parameter list:")
            self.discrim_model.print_parameter_list()

    def on_epoch_begin(self):
        """Called at the beginning of an epoch.

        Updates all schedulers for both generator and discriminator.
        """
        super().on_epoch_begin()

        for sch in [self.vc_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None:
                sch.on_epoch_begin(self.cur_epoch, epoch_updates=self.save_steps)

        for sch in [self.vc_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_epoch_begin(self.cur_epoch)

    def on_epoch_end(self, logs):
        """Called at the end of an epoch.

        Steps schedulers unless currently in the SWA phase.
        """
        super().on_epoch_end(logs)
        if self.do_swa and self.cur_step >= self.swa_start:
            return

        for sch in [self.vc_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None:
                sch.on_epoch_end(self.cur_epoch)

        for sch in [self.vc_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_epoch_end(self.cur_epoch)

    def on_swa_epoch_begin(self):
        """Called at the beginning of an SWA epoch.

        Swaps the current VC model with the averaged SWA model.
        """
        super().on_swa_epoch_begin()
        self.vc_model = self.swa_vc_model.module

    def on_swa_epoch_end(self, logs):
        super().on_swa_epoch_end(logs)

    def on_training_loop_begin(self):
        """Sets models to training mode before beginning the training loop."""
        self.vc_model.train()
        self.discrim_model.train()
        self.audio_feats.train()
        self.xvector_model.eval()

    def on_val_loop_begin(self):
        """Sets models to evaluation mode before starting validation."""
        self.vc_model.eval()
        self.discrim_model.eval()
        self.audio_feats.eval()
        self.xvector_model.eval()
        self.cur_val_log_samples = 0

    def preprocess_train_data(self, batch_data):
        """Prepares and renames training batch data into a standardized format.

        Returns:
            Tuple[int, Dict[str, Tensor]]: Batch size and processed batch.
        """
        output_batch_data = {
            "id": batch_data["id"],
            "source_audios": batch_data[self.input_audio_key],
            "source_audio_lengths": batch_data[f"{self.input_audio_key}_lengths"],
            "target_audios": batch_data[self.target_audio_key],
            "target_audio_lengths": batch_data[f"{self.target_audio_key}_lengths"],
        }
        batch_size = output_batch_data["source_audios"].size(0)
        return batch_size, output_batch_data

    def preprocess_val_data(self, batch_data):
        return self.preprocess_train_data(batch_data)

    def training_step(
        self, batch_idx: int, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """Performs the forward and backward passes for both discriminator and generator.

        Handles discriminator training first with real/fake inputs, then updates
        the generator using adversarial and auxiliary losses.

        Returns:
            OrderedDict[str, float]: A dictionary of computed metrics.
        """
        batch_size, batch_data = self.preprocess_train_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)
        ############################
        # 1. Discriminator Forward #
        ############################
        self.discrim_model.set_train_mode(self.discrim_train_mode)
        input_audios, input_lengths = (
            batch_data[f"source_audios"],
            batch_data[f"source_audio_lengths"],
        )

        target_audios, target_lengths = (
            batch_data[f"target_audios"],
            batch_data[f"target_audio_lengths"],
        )
        # print(batch_data, flush=True)
        with torch.no_grad():
            target_audios_matched, target_matched_lengths = (
                self.vc_model.get_target_matching_output(
                    target_audios, target_lengths, input_audios.shape[-1]
                )
            )
            target_audios, slice_start_idxs = rand_slice_audio_segments(
                target_audios_matched,
                target_matched_lengths,
                self.gen_segment_duration,
                self.vc_model.output_sample_frequency,
            )
            # target_audios, slice_start_idxs = rand_slice_audio_segments(
            #     target_audios,
            #     target_lengths,
            #     self.gen_segment_duration,
            #     self.vc_model.output_sample_frequency,
            # )

        with (
            torch.no_grad(),
            amp.autocast(
                enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
            ),
        ):
            xvector_output = self.xvector_model(
                input_audios,
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            speaker_feats = xvector_output.xvector

        with amp.autocast(
            enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
        ):
            vc_output = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.RECONS,
                slice_start_idxs=slice_start_idxs,
                slice_segment_length=int(
                    self.gen_segment_duration * self.vc_model.output_sample_frequency
                ),
            )

            y_real, _ = self.discrim_model(
                target_audios,
            )
            y_gen, _ = self.discrim_model(vc_output.gen_audio.detach())

        loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
            self.discrim_adv_loss(y_gen, y_real)
        )
        loss_discrim = loss_discrim / self.grad_acc_steps
        self.grad_scaler.scale(loss_discrim).backward()

        ########################
        # 2. Generator Forward #
        ########################
        self.discrim_model.set_train_mode(AudioDiscriminatorTrainMode.FROZEN)
        with amp.autocast(
            enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
        ):
            y_real, fmaps_real = self.discrim_model(target_audios)
            y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
            with torch.no_grad():
                mel_feats_real, mels_feats_real_lengths = self.audio_feats(
                    target_audios
                )

            mel_feats_gen, mel_feats_gen_lengths = self.audio_feats(
                vc_output.gen_audio.squeeze(1)
            )

        loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
        loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
        loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
        loss_kldiv = vc_output.kldiv_loss
        loss_gen = (
            self.loss_gen_adv_weight * loss_gen_adv
            + self.loss_fm_weight * loss_fm
            + self.loss_mel_weight * loss_mel
            + self.loss_kl_weight * loss_kldiv
        ) / self.grad_acc_steps

        self.grad_scaler.scale(loss_gen).backward()

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total"] = loss_discrim.item() * self.grad_acc_steps
        batch_metrics["loss_gen/total"] = loss_gen.item() * self.grad_acc_steps
        batch_metrics["loss_gen/mel"] = loss_mel.item()
        batch_metrics["loss_gen/kldiv"] = loss_kldiv.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv"] = loss_gen_adv.item()
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv/{i}"] = loss

        return batch_size, batch_metrics

    def validation_step(self, batch_idx: int, batch_data: Dict[str, Any]):
        """Runs a forward pass through the generator and discriminator during validation.

        Logs spectrograms and audio samples, and computes validation losses.

        Returns:
            Tuple[int, Dict[str, float]]: Batch size and metrics.
        """
        batch_size, batch_data = self.preprocess_val_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)

        input_audios, input_lengths = (
            batch_data[f"source_audios"],
            batch_data[f"source_audio_lengths"],
        )

        target_audios, target_lengths = (
            batch_data[f"target_audios"],
            batch_data[f"target_audio_lengths"],
        )
        with torch.no_grad():
            target_audios, target_lengths = self.vc_model.get_target_matching_output(
                target_audios, target_lengths, input_audios.shape[-1]
            )
            target_audios_sliced, slice_start_idxs = rand_slice_audio_segments(
                target_audios,
                target_lengths,
                self.gen_segment_duration,
                self.vc_model.output_sample_frequency,
            )

        with amp.autocast(
            enabled=self.use_amp, dtype=self.amp_dtype, device_type="cuda"
        ):
            xvector_output = self.xvector_model(
                input_audios,
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            speaker_feats = xvector_output.xvector
            vc_output = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.VC,
                # slice_start_idxs=slice_start_idxs,
                # slice_segment_length=int(
                #     self.gen_segment_duration * self.vc_model.output_sample_frequency
                # ),
            )
            if (
                vc_output.z.shape[1]
                != self.vc_model.hf_feats.max_out_length(input_audios.shape[-1])
                or target_audios.shape[1] != vc_output.gen_audio.shape[-1]
            ):
                print(
                    "sliced",
                    input_audios.shape,
                    input_lengths,
                    target_audios.shape,
                    target_lengths,
                    target_audios_sliced.shape,
                    vc_output.gen_audio.shape,
                    vc_output.z.shape,
                    slice_start_idxs,
                    int(
                        self.gen_segment_duration
                        * self.vc_model.output_sample_frequency
                    ),
                    self.vc_model.hf_feats.max_out_length(input_audios.shape[-1]),
                    flush=True,
                )
            gen_audios_sliced = slice_segments(
                vc_output.gen_audio.squeeze(1),
                slice_start_idxs,
                int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
            )
            y_real, fmaps_real = self.discrim_model(target_audios_sliced)
            # y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
            y_gen, fmaps_gen = self.discrim_model(gen_audios_sliced)
            mel_feats_real, _ = self.audio_feats(target_audios)
            mel_feats_gen, _ = self.audio_feats(vc_output.gen_audio.squeeze(1))

        loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
            self.discrim_adv_loss(y_gen, y_real)
        )
        loss_adv_gen, losses_gen_adv = self.gen_adv_loss(y_gen)
        loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
        loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
        loss_gen = (
            self.loss_gen_adv_weight * loss_adv_gen
            + self.loss_fm_weight * loss_fm
            + self.loss_mel_weight * loss_mel
        )

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total"] = loss_discrim.item()
        batch_metrics["loss_gen/total"] = loss_gen.item()
        batch_metrics["loss_gen/mel"] = loss_mel.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv"] = loss_adv_gen.item()
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv/{i}"] = loss

        num_log_samples = min(
            self.num_val_log_samples - self.cur_val_log_samples, batch_size
        )
        for i in range(num_log_samples):
            _id = batch_data["id"][i]
            self.loggers.log_audio(
                f"audios_target/{_id}",
                target_audios[i],
                sample_freq=self.vc_model.output_sample_frequency,
            )
            self.loggers.log_audio(
                f"audios_generated/{_id}",
                vc_output.gen_audio[i, 0],
                sample_freq=self.vc_model.output_sample_frequency,
            )
            self.loggers.log_spectrogram(
                f"log_mel_fbanks_target/{_id}", mel_feats_real[i]
            )
            self.loggers.log_spectrogram(
                f"log_mel_fbanks_generated/{_id}", mel_feats_gen[i]
            )

        self.cur_val_log_samples += num_log_samples

        return batch_size, batch_metrics

    def update_swa_model(self):
        """Updates the SWA model parameters and learning rate scheduler if applicable."""
        if (
            self.do_swa
            and self.cur_step >= self.swa_start
            and self.cur_step % self.swa_update_steps == 0
        ):
            self.in_swa = True
            self.swa_vc_model.update_parameters(self.vc_model)
            self.swa_vc_scheduler.step()

    def zero_grad_optimizers(self):
        """Zeros the gradients for both generator and discriminator optimizers."""
        self.vc_optimizer.zero_grad()
        self.discrim_optimizer.zero_grad()

    def get_lrs(self):
        """Returns a dictionary of learning rates for all optimizers."""
        vc_lrs = self._get_lrs(self.vc_optimizer)
        discrim_lrs = self._get_lrs(self.discrim_optimizer)
        lrs = {f"vc_{k}": v for k, v in vc_lrs.items()}
        discrim_lrs = {f"discrim_{k}": v for k, v in discrim_lrs.items()}
        lrs.update(discrim_lrs)
        return lrs

    def get_wds(self):
        """Returns a dictionary of weight decay values for all optimizers."""
        vc_wds = self._get_wds(self.vc_optimizer, self.vc_wd_scheduler)
        discrim_wds = self._get_wds(self.discrim_optimizer, self.discrim_wd_scheduler)
        wds = {f"vc_{k}": v for k, v in vc_wds.items()}
        wds.update({f"discrim_{k}": v for k, v in discrim_wds.items()})
        return wds

    def models_have_bn(self):
        """Checks if the generator model has any batch normalization layers."""
        return self.vc_model.has_batchnorms()

    def update_models(self):
        """Steps optimizers and schedulers for both generator and discriminator.

        Also clips gradients and returns logs for gradient norms.

        Returns:
            Dict[str, float]: Gradient norm logs.
        """

        for sch in [self.vc_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None and not self.in_swa:
                sch.on_opt_step()

        for sch in [self.vc_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_opt_step()

        vc_grad_norm = self._update_model_by_optim(
            self.vc_model,
            self.vc_optimizer,
            self.grad_clip,
            self.grad_clip_norm,
            self.use_amp,
            self.grad_scaler,
        )
        discrim_grad_norm = self._update_model_by_optim(
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_grad_clip,
            self.grad_clip_norm,
            self.use_amp,
            self.grad_scaler,
        )
        self.grad_scaler.update()

        logs = {"grad_norm/vc": vc_grad_norm, "grad_norm/discrim": discrim_grad_norm}
        return logs

    def save_checkpoint(self, logs=None):
        """Saves current training state to disk, including both models and optionally SWA.

        Args:
            logs (Optional[Dict[str, Any]]): Logging metrics to include in the checkpoint.
        """
        if self.ddp and (
            self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
        ):
            # Not sure what this does, just copying from the example in
            # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
            # Check the checkpointing in the case of the OSS optimizer
            # Memory usage could spill over from there
            # optimizer = cast(OSS, optimizer)
            self.vc_optimizer.consolidate_state_dict()
            self.discrim_optimizer.consolidate_state_dict()

        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.vc_model,
            self.vc_optimizer,
            self.vc_lr_scheduler,
            self.vc_wd_scheduler,
            self.swa_vc_model,
            self.swa_vc_scheduler,
            logs=logs,
        )

        self.save_model_checkpoint_to_file("vc_model", checkpoint)

        checkpoint = self.model_checkpoint(
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_lr_scheduler,
            self.discrim_wd_scheduler,
            logs=logs,
        )

        self.save_model_checkpoint_to_file("discrim_model", checkpoint)

    def save_swa_model(self, logs=None):
        """Saves the final SWA-averaged generator model to disk.

        Args:
            logs (Optional[Dict[str, Any]]): Logging metrics to include in the checkpoint.
        """
        if self.rank != 0:
            return

        checkpoint = self.model_checkpoint(
            self.vc_model,
            self.vc_optimizer,
            self.vc_lr_scheduler,
            self.vc_wd_scheduler,
            self.swa_vc_model,
            self.swa_vc_scheduler,
            logs=logs,
        )
        checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
        del checkpoint["swa_model_state_dict"]
        file_path = "%s/swa_vc_model_ep%04d_%010d.pth" % (
            self.exp_path,
            self.cur_epoch,
            self.cur_step,
        )
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, epoch, step):
        """Loads training state from checkpoint files for both generator and discriminator.

        Args:
            epoch (int): Epoch number of checkpoint.
            step (int): Step number of checkpoint.

        Returns:
            Optional[Dict[str, Any]]: Logs saved with the checkpoint, if any.
        """
        checkpoint = self.load_model_checkpoint_from_file("vc_model", epoch, step)
        logs = self._load_vars_from_checkpoint(checkpoint)
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.vc_model,
            self.vc_optimizer,
            self.vc_lr_scheduler,
            self.vc_wd_scheduler,
            self.swa_vc_model,
            self.swa_vc_scheduler,
        )
        checkpoint = self.load_model_checkpoint_from_file("discrim_model", epoch, step)
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_lr_scheduler,
            self.discrim_wd_scheduler,
        )
        return logs

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters the provided keyword arguments to retain only those valid for the FreeVCTrainer constructor.

        Returns:
            dict: A dictionary of filtered arguments applicable to FreeVCTrainer.
        """
        args = filter_func_args(FreeVCTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_optim_args(parser, prefix=None):
        """
        Adds command-line arguments for generator and discriminator optimizers and schedulers.

        Args:
            parser (ArgumentParser): Argument parser instance to which arguments are added.
            prefix (str, optional): Optional namespace prefix to encapsulate arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        OF.add_class_args(parser, prefix="vc_optim")
        LRSF.add_class_args(parser, prefix="vc_lrsched")
        WDSF.add_class_args(parser, prefix="vc_wdsched")
        OF.add_class_args(parser, prefix="discrim_optim")
        LRSF.add_class_args(parser, prefix="discrim_lrsched")
        WDSF.add_class_args(parser, prefix="discrim_wdsched")
        parser.add_argument(
            "--discrim-grad-clip",
            default=0.0,
            type=float,
            help="Gradient clipping value for the discriminator.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_train_modes_args(parser):
        """
        Adds command-line arguments for generator and discriminator optimizers and schedulers.

        Args:
            parser (ArgumentParser): Argument parser instance to which arguments are added.
            prefix (str, optional): Optional namespace prefix to encapsulate arguments.
        """
        train_modes = FreeVCTrainMode.choices()
        parser.add_argument(
            "--vc-train-mode",
            default=FreeVCTrainMode.HF_FEATS_FROZEN_NOGRAD.value,
            choices=train_modes,
            help=(
                f"Training mode for the generator. "
                f"Available options: {train_modes}."
            ),
        )

    @staticmethod
    def add_io_keys_args(parser, prefix=None):
        """
        Adds command-line arguments to specify batch dictionary keys for input and target audio.

        Args:
            parser (ArgumentParser): Argument parser to which arguments are added.
            prefix (str, optional): Optional namespace prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--input-audio-key",
            default="audio",
            help="Key used to access source audio in the batch dictionary.",
        )
        parser.add_argument(
            "--target-audio-key",
            default="audio",
            help="Key used to access target audio in the batch dictionary.",
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_loss_weights_args(parser, prefix=None, skip=set()):
        """
        Adds command-line arguments to configure loss weights for the generator.

        Args:
            parser (ArgumentParser): Argument parser to which arguments are added.
            prefix (str, optional): Optional namespace prefix.
            skip (set): Set of argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "loss_mel_weight" not in skip:
            parser.add_argument(
                "--loss-mel-weight",
                default=45.0,
                type=float,
                help="Weight for the mel-spectrogram reconstruction loss.",
            )
        parser.add_argument(
            "--loss-kl-weight",
            default=1.0,
            type=float,
            help="Weight for the KL divergence loss (used in VAE components).",
        )
        parser.add_argument(
            "--loss-fm-weight",
            default=2.0,
            type=float,
            help="Weight for the discriminator feature-matching loss.",
        )
        parser.add_argument(
            "--loss-gen-adv-weight",
            default=1.0,
            type=float,
            help="Weight for the adversarial generator loss.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        """
        Adds all FreeVCTrainer-related arguments to the parser, including trainer, optimizer, I/O, and loss configuration.

        Args:
            parser (ArgumentParser): Argument parser to which arguments are added.
            prefix (str, optional): Optional prefix to namespace all arguments.
            skip (set): Set of argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        TorchTrainerBase.add_class_args(parser)
        AudioFeatsMVN.add_class_args(parser, prefix="audio_feats")
        FreeVCTrainer.add_optim_args(parser)
        FreeVCTrainer.add_io_keys_args(parser)
        FreeVCTrainer.add_train_modes_args(parser)
        FreeVCTrainer.add_loss_weights_args(parser)
        parser.add_argument(
            "--gen-segment-duration",
            default=0.64,
            type=float,
            help="Duration (in seconds) of the audio segments used as input to the discrimator to be used during training and validation.",
        )
        parser.add_argument(
            "--num-val-log-samples",
            default=10,
            type=int,
            help="Number of samples to log during validation (audio + spectrogram).",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))


# import argparse
# import itertools
# import json
# import math
# import os

# import commons
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import utils
# from data_utils import (
#     DistributedBucketSampler,
#     TextAudioSpeakerCollate,
#     TextAudioSpeakerLoader,
# )
# from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
# from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
# from models import MultiPeriodDiscriminator, SynthesizerTrn
# from torch import nn, optim
# from torch.cuda.amp import GradScaler, autocast
# from torch.nn import functional as F
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.benchmark = True
# global_step = 0
# # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


# def main():
#     """Assume Single Node Multi GPUs Training Only"""
#     assert torch.cuda.is_available(), "CPU training is not allowed."
#     hps = utils.get_hparams()

#     n_gpus = torch.cuda.device_count()
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = hps.train.port

#     mp.spawn(
#         run,
#         nprocs=n_gpus,
#         args=(
#             n_gpus,
#             hps,
#         ),
#     )


# def run(rank, n_gpus, hps):
#     global global_step
#     if rank == 0:
#         logger = utils.get_logger(hps.model_dir)
#         logger.info(hps)
#         utils.check_git_hash(hps.model_dir)
#         writer = SummaryWriter(log_dir=hps.model_dir)
#         writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

#     dist.init_process_group(
#         backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
#     )
#     torch.manual_seed(hps.train.seed)
#     torch.cuda.set_device(rank)

#     train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
#     train_sampler = DistributedBucketSampler(
#         train_dataset,
#         hps.train.batch_size,
#         [32, 300, 400, 500, 600, 700, 800, 900, 1000],
#         num_replicas=n_gpus,
#         rank=rank,
#         shuffle=True,
#     )
#     collate_fn = TextAudioSpeakerCollate(hps)
#     train_loader = DataLoader(
#         train_dataset,
#         num_workers=8,
#         shuffle=False,
#         pin_memory=True,
#         collate_fn=collate_fn,
#         batch_sampler=train_sampler,
#     )
#     if rank == 0:
#         eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
#         eval_loader = DataLoader(
#             eval_dataset,
#             num_workers=8,
#             shuffle=True,
#             batch_size=hps.train.batch_size,
#             pin_memory=False,
#             drop_last=False,
#             collate_fn=collate_fn,
#         )

#     net_g = SynthesizerTrn(
#         hps.data.filter_length // 2 + 1,
#         hps.train.segment_size // hps.data.hop_length,
#         **hps.model,
#     ).cuda(rank)
#     net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
#     optim_g = torch.optim.AdamW(
#         net_g.parameters(),
#         hps.train.learning_rate,
#         betas=hps.train.betas,
#         eps=hps.train.eps,
#     )
#     optim_d = torch.optim.AdamW(
#         net_d.parameters(),
#         hps.train.learning_rate,
#         betas=hps.train.betas,
#         eps=hps.train.eps,
#     )
#     net_g = DDP(net_g, device_ids=[rank])  # , find_unused_parameters=True)
#     net_d = DDP(net_d, device_ids=[rank])

#     try:
#         _, _, _, epoch_str = utils.load_checkpoint(
#             utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
#         )
#         _, _, _, epoch_str = utils.load_checkpoint(
#             utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
#         )
#         global_step = (epoch_str - 1) * len(train_loader)
#     except:
#         epoch_str = 1
#         global_step = 0

#     scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
#         optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
#     )
#     scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
#         optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
#     )

#     scaler = GradScaler(enabled=hps.train.fp16_run)

#     for epoch in range(epoch_str, hps.train.epochs + 1):
#         if rank == 0:
#             train_and_evaluate(
#                 rank,
#                 epoch,
#                 hps,
#                 [net_g, net_d],
#                 [optim_g, optim_d],
#                 [scheduler_g, scheduler_d],
#                 scaler,
#                 [train_loader, eval_loader],
#                 logger,
#                 [writer, writer_eval],
#             )
#         else:
#             train_and_evaluate(
#                 rank,
#                 epoch,
#                 hps,
#                 [net_g, net_d],
#                 [optim_g, optim_d],
#                 [scheduler_g, scheduler_d],
#                 scaler,
#                 [train_loader, None],
#                 None,
#                 None,
#             )
#         scheduler_g.step()
#         scheduler_d.step()


# def train_and_evaluate(
#     rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
# ):

#     net_g, net_d = nets
#     optim_g, optim_d = optims
#     scheduler_g, scheduler_d = schedulers
#     train_loader, eval_loader = loaders
#     if writers is not None:
#         writer, writer_eval = writers

#     train_loader.batch_sampler.set_epoch(epoch)
#     global global_step

#     net_g.train()
#     net_d.train()
#     for batch_idx, items in enumerate(train_loader):
#         if hps.model.use_spk:
#             c, spec, y, spk = items
#             g = spk.cuda(rank, non_blocking=True)
#         else:
#             c, spec, y = items
#             g = None
#         spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
#         c = c.cuda(rank, non_blocking=True)
#         mel = spec_to_mel_torch(
#             spec,
#             hps.data.filter_length,
#             hps.data.n_mel_channels,
#             hps.data.sampling_rate,
#             hps.data.mel_fmin,
#             hps.data.mel_fmax,
#         )

#         with autocast(enabled=hps.train.fp16_run):
#             y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
#                 c, spec, g=g, mel=mel
#             )

#             y_mel = commons.slice_segments(
#                 mel, ids_slice, hps.train.segment_size // hps.data.hop_length
#             )
#             y_hat_mel = mel_spectrogram_torch(
#                 y_hat.squeeze(1),
#                 hps.data.filter_length,
#                 hps.data.n_mel_channels,
#                 hps.data.sampling_rate,
#                 hps.data.hop_length,
#                 hps.data.win_length,
#                 hps.data.mel_fmin,
#                 hps.data.mel_fmax,
#             )
#             y = commons.slice_segments(
#                 y, ids_slice * hps.data.hop_length, hps.train.segment_size
#             )  # slice

#             # Discriminator
#             y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
#             with autocast(enabled=False):
#                 loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
#                     y_d_hat_r, y_d_hat_g
#                 )
#                 loss_disc_all = loss_disc

#         optim_d.zero_grad()
#         scaler.scale(loss_disc_all).backward()
#         scaler.unscale_(optim_d)
#         grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
#         scaler.step(optim_d)

#         with autocast(enabled=hps.train.fp16_run):
#             # Generator
#             y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
#             with autocast(enabled=False):
#                 loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
#                 loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
#                 loss_fm = feature_loss(fmap_r, fmap_g)
#                 loss_gen, losses_gen = generator_loss(y_d_hat_g)
#                 loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
#         optim_g.zero_grad()
#         scaler.scale(loss_gen_all).backward()
#         scaler.unscale_(optim_g)
#         grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
#         scaler.step(optim_g)
#         scaler.update()

#         if rank == 0:
#             if global_step % hps.train.log_interval == 0:
#                 lr = optim_g.param_groups[0]["lr"]
#                 losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
#                 logger.info(
#                     "Train Epoch: {} [{:.0f}%]".format(
#                         epoch, 100.0 * batch_idx / len(train_loader)
#                     )
#                 )
#                 logger.info([x.item() for x in losses] + [global_step, lr])

#                 scalar_dict = {
#                     "loss/g/total": loss_gen_all,
#                     "loss/d/total": loss_disc_all,
#                     "learning_rate": lr,
#                     "grad_norm_d": grad_norm_d,
#                     "grad_norm_g": grad_norm_g,
#                 }
#                 scalar_dict.update(
#                     {"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl}
#                 )

#                 scalar_dict.update(
#                     {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
#                 )
#                 scalar_dict.update(
#                     {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
#                 )
#                 scalar_dict.update(
#                     {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
#                 )
#                 image_dict = {
#                     "slice/mel_org": utils.plot_spectrogram_to_numpy(
#                         y_mel[0].data.cpu().numpy()
#                     ),
#                     "slice/mel_gen": utils.plot_spectrogram_to_numpy(
#                         y_hat_mel[0].data.cpu().numpy()
#                     ),
#                     "all/mel": utils.plot_spectrogram_to_numpy(
#                         mel[0].data.cpu().numpy()
#                     ),
#                 }
#                 utils.summarize(
#                     writer=writer,
#                     global_step=global_step,
#                     images=image_dict,
#                     scalars=scalar_dict,
#                 )

#             if global_step % hps.train.eval_interval == 0:
#                 evaluate(hps, net_g, eval_loader, writer_eval)
#                 utils.save_checkpoint(
#                     net_g,
#                     optim_g,
#                     hps.train.learning_rate,
#                     epoch,
#                     os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
#                 )
#                 utils.save_checkpoint(
#                     net_d,
#                     optim_d,
#                     hps.train.learning_rate,
#                     epoch,
#                     os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
#                 )
#         global_step += 1

#     if rank == 0:
#         logger.info("====> Epoch: {}".format(epoch))


# def evaluate(hps, generator, eval_loader, writer_eval):
#     generator.eval()
#     with torch.no_grad():
#         for batch_idx, items in enumerate(eval_loader):
#             if hps.model.use_spk:
#                 c, spec, y, spk = items
#                 g = spk[:1].cuda(0)
#             else:
#                 c, spec, y = items
#                 g = None
#             spec, y = spec[:1].cuda(0), y[:1].cuda(0)
#             c = c[:1].cuda(0)
#             break
#         mel = spec_to_mel_torch(
#             spec,
#             hps.data.filter_length,
#             hps.data.n_mel_channels,
#             hps.data.sampling_rate,
#             hps.data.mel_fmin,
#             hps.data.mel_fmax,
#         )
#         y_hat = generator.module.infer(c, g=g, mel=mel)

#         y_hat_mel = mel_spectrogram_torch(
#             y_hat.squeeze(1).float(),
#             hps.data.filter_length,
#             hps.data.n_mel_channels,
#             hps.data.sampling_rate,
#             hps.data.hop_length,
#             hps.data.win_length,
#             hps.data.mel_fmin,
#             hps.data.mel_fmax,
#         )
#     image_dict = {
#         "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
#         "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
#     }
#     audio_dict = {"gen/audio": y_hat[0], "gt/audio": y[0]}
#     utils.summarize(
#         writer=writer_eval,
#         global_step=global_step,
#         images=image_dict,
#         audios=audio_dict,
#         audio_sampling_rate=hps.data.sampling_rate,
#     )
#     generator.train()

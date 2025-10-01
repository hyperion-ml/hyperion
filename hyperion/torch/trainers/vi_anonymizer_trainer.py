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
from typing import Any, Dict, List, Optional, Union

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..loggers import LoggerList
from ..losses import (
    AudioDiscriminatorAdvLoss,
    AudioGeneratorAdvLoss,
    ContrastiveLoss,
    FeatureMatchingLoss,
    MultiResolutionFilterBankLoss,
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
from .freevc_trainer import FreeVCTrainer
from .torch_trainer_base import AMPDType, DDPType, TorchTrainerBase


class VIAnonymizerTrainer(FreeVCTrainer):
    """Trainer for Variational Inference Anonymizer (VIAnonymizer).

    This trainer handles the training of voice anonymization models used in adversarial training setups.
    It supports mixed precision, SWA, DDP, and advanced logging through TensorBoard and W&B.

    Attributes:
        vc_model: Generator model for voice conversion.
        discrim_model: Discriminator model for adversarial training.
        xvector_model: Pretrained speaker embedding model.
        audio_feats: Feature extractor for audio (e.g., log-mel spectrograms).
        speaker_contrastive_loss: Loss function for speaker contrastive.
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
        loss_mrfb_log_mag_weight: float = 15.0,
        loss_mrfb_conv_weight: float = 15.0,
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
        mrfb_loss: Union[MultiResolutionFilterBankLoss, Dict[str, Any]],
        audio_feats: AudioFeatsMVN,
        speaker_contrastive_loss: Dict[str, Any],
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
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: int = 1e-3,
        swa_anneal_steps: int = 50000,
        input_audio_key="audio",
        target_audio_key="audio",
        speaker_key="speaker",
        loss_mel_weight: float = 0.0,
        loss_mrfb_log_mag_weight: float = 45.0,
        loss_mrfb_conv_weight: float = 1.0,
        loss_kl_weight: float = 1.0,
        loss_gen_adv_weight: float = 1.0,
        loss_fm_weight: float = 1.0,
        loss_vc_adv_weight: float = 1.0,
        loss_speaker_contrastive_weight: float = 1.0,
        vc_losses_warmup_steps: int = 50000,
        vc_losses_warmup_start_step: int = 50000,
        gen_segment_duration: float = 0.64,
        num_val_log_samples: int = 10,
    ):

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)
        self.speaker_contrastive_loss = ContrastiveLoss(**speaker_contrastive_loss)
        self.loss_mrfb_log_mag_weight = loss_mrfb_log_mag_weight
        self.loss_mrfb_conv_weight = loss_mrfb_conv_weight
        self.loss_vc_adv_weight = loss_vc_adv_weight
        self.loss_speaker_contrastive_weight = loss_speaker_contrastive_weight
        self.speaker_key = speaker_key

        self.vc_losses_warmup_steps = vc_losses_warmup_steps
        self.vc_losses_warmup_start_step = vc_losses_warmup_start_step
        self.speaker_contrastive_loss.to(self.device)

        if isinstance(mrfb_loss, dict):
            self.mrfb_loss = MultiResolutionFilterBankLoss(**mrfb_loss).to(self.device)
        else:
            self.mrfb_loss = mrfb_loss.to(self.device)

        if self.rank == 0:
            logging.info(f"MRFB Loss:\n{self.mrfb_loss}")

    def on_epoch_begin(self):
        """Called at the beginning of an epoch.

        Updates all schedulers for both generator and discriminator.
        """
        super().on_epoch_begin()

    def on_epoch_end(self, logs):
        """Called at the end of an epoch.

        Steps schedulers unless currently in the SWA phase.
        """
        super().on_epoch_end(logs)

    def on_train_loop_begin(self):
        """Sets models to training mode before beginning the training loop."""
        super().on_train_loop_begin()
        self.speaker_contrastive_loss.train()

    def on_val_loop_begin(self):
        """Sets models to evaluation mode before starting validation."""
        super().on_val_loop_begin()
        self.speaker_contrastive_loss.eval()

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
            "speaker": batch_data.get(self.speaker_key, None),
        }
        batch_size = output_batch_data["source_audios"].size(0)
        return batch_size, output_batch_data

    # def train_forward_backward_1(self, batch_data):
    #     """Performs the forward and backward passes for both discriminator and generator.

    #     Handles discriminator training first with real/fake inputs, then updates
    #     the generator using adversarial and auxiliary losses.

    #     Returns:
    #         OrderedDict[str, float]: A dictionary of computed metrics.
    #     """
    #     self.speaker_contrastive_loss.update(self.cur_step)
    #     ############################
    #     # 1. Discriminator Forward #
    #     ############################
    #     self.discrim_model.set_train_mode(self.discrim_train_mode)
    #     input_audios, input_lengths = (
    #         batch_data[f"source_audios"],
    #         batch_data[f"source_audio_lengths"],
    #     )

    #     target_audios, target_lengths = (
    #         batch_data[f"target_audios"],
    #         batch_data[f"target_audio_lengths"],
    #     )
    #     # print(batch_data, flush=True)
    #     with torch.no_grad():
    #         target_audios_matched, target_matched_lengths = (
    #             self.vc_model.get_target_matching_output(
    #                 target_audios, target_lengths, input_audios.shape[-1]
    #             )
    #         )
    #         target_audios, slice_start_idxs = rand_slice_audio_segments(
    #             target_audios_matched,
    #             target_matched_lengths,
    #             self.gen_segment_duration,
    #             self.vc_model.output_sample_frequency,
    #         )

    #     with torch.no_grad(), amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         xvector_output = self.xvector_model(
    #             input_audios,
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         speaker_feats = xvector_output.xvector

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         feats, feat_lengths = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=None,
    #             mode=FreeVCFwdMode.FEATS_ONLY,
    #         )
    #         vc_output = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.RECONS,
    #             slice_start_idxs=slice_start_idxs,
    #             slice_segment_length=int(
    #                 self.gen_segment_duration * self.vc_model.output_sample_frequency
    #             ),
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )

    #         y_real, _ = self.discrim_model(
    #             target_audios,
    #         )
    #         y_gen, _ = self.discrim_model(vc_output.gen_audio.detach())

    #     loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
    #         self.discrim_adv_loss(y_gen, y_real)
    #     )
    #     loss_discrim = loss_discrim / self.grad_acc_steps
    #     self.grad_scaler.scale(loss_discrim).backward()

    #     #######################################
    #     # 2. Generator Forward Reconstruction #
    #     #######################################
    #     self.discrim_model.set_train_mode(AudioDiscriminatorTrainMode.FROZEN)
    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         y_real, fmaps_real = self.discrim_model(target_audios)
    #         y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
    #         with torch.no_grad():
    #             mel_feats_real, mels_feats_real_lengths = self.audio_feats(
    #                 target_audios
    #             )

    #         mel_feats_gen, mel_feats_gen_lengths = self.audio_feats(
    #             vc_output.gen_audio.squeeze(1)
    #         )

    #     loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
    #     loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
    #     loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
    #     loss_kldiv = vc_output.kldiv_loss
    #     loss_gen_recons = (
    #         self.loss_gen_adv_weight * loss_gen_adv
    #         + self.loss_fm_weight * loss_fm
    #         + self.loss_mel_weight * loss_mel
    #         + self.loss_kl_weight * loss_kldiv
    #     ) / self.grad_acc_steps

    #     ###########################
    #     # 3. Generator Forward VC #
    #     ###########################
    #     with torch.no_grad():
    #         rand_perm = torch.randperm(len(speaker_feats), device=speaker_feats.device)
    #         speaker_feats = speaker_feats[rand_perm]

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         vc_output = self.vc_model(
    #             source_audios=None,
    #             source_audio_lengths=None,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.VC,
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )
    #         vc_audios_sliced = slice_segments(
    #             vc_output.gen_audio.squeeze(1),
    #             slice_start_idxs,
    #             int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
    #         ).unsqueeze(1)
    #         y_gen, fmaps_gen = self.discrim_model(vc_audios_sliced)
    #         xvector_output = self.xvector_model(
    #             vc_output.gen_audio.squeeze(1),
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         gen_speaker_feats = xvector_output.xvector

    #     loss_gen_adv_vc, losses_gen_adv_vc = self.gen_adv_loss(y_gen)
    #     loss_speaker_contrastive = self.speaker_contrastive_loss(
    #         gen_speaker_feats, speaker_feats
    #     )
    #     loss_gen_vc = (
    #         self.loss_vc_adv_weight * loss_gen_adv_vc
    #         + self.loss_speaker_contrastive_weight * loss_speaker_contrastive
    #     ) / self.grad_acc_steps

    #     if self.cur_step < self.vc_losses_warmup_steps:
    #         vc_weight = (
    #             1 - math.cos(math.pi * self.cur_step / self.vc_losses_warmup_steps)
    #         ) / 2
    #     else:
    #         vc_weight = 1.0

    #     loss_gen_total = loss_gen_recons + vc_weight * loss_gen_vc
    #     self.grad_scaler.scale(loss_gen_total).backward()

    #     batch_metrics = ODict()
    #     batch_metrics["loss_discrim/total"] = loss_discrim.item() * self.grad_acc_steps
    #     batch_metrics["loss_gen/total_recons"] = (
    #         loss_gen_recons.item() * self.grad_acc_steps
    #     )
    #     batch_metrics["loss_gen/total_vc"] = loss_gen_vc.item() * self.grad_acc_steps
    #     batch_metrics["loss_gen/mel"] = loss_mel.item()
    #     batch_metrics["loss_gen/kldiv"] = loss_kldiv.item()
    #     batch_metrics["loss_gen/fm"] = loss_fm.item()
    #     batch_metrics["loss_gen/adv_recons"] = loss_gen_adv.item()
    #     batch_metrics["loss_gen/adv_vc"] = loss_gen_adv_vc.item()
    #     batch_metrics["loss_gen/speaker_contrastive"] = loss_speaker_contrastive.item()
    #     for i, loss in enumerate(losses_discrim_adv_gen):
    #         batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

    #     for i, loss in enumerate(losses_discrim_adv_real):
    #         batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv):
    #         batch_metrics[f"loss_gen_adv_recons/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv_vc):
    #         batch_metrics[f"loss_gen_adv_vc/{i}"] = loss

    #     return batch_metrics

    # def train_forward_backward_2(self, batch_data):
    #     """Performs the forward and backward passes for both discriminator and generator.

    #     Handles discriminator training first with real/fake inputs, then updates
    #     the generator using adversarial and auxiliary losses.

    #     Returns:
    #         OrderedDict[str, float]: A dictionary of computed metrics.
    #     """
    #     self.speaker_contrastive_loss.update(self.cur_step)
    #     ############################
    #     # 1. Discriminator Forward #
    #     ############################
    #     self.discrim_model.set_train_mode(self.discrim_train_mode)
    #     input_audios, input_lengths = (
    #         batch_data[f"source_audios"],
    #         batch_data[f"source_audio_lengths"],
    #     )

    #     target_audios, target_lengths = (
    #         batch_data[f"target_audios"],
    #         batch_data[f"target_audio_lengths"],
    #     )
    #     # print(batch_data, flush=True)
    #     with torch.no_grad():
    #         target_audios_matched, target_matched_lengths = (
    #             self.vc_model.get_target_matching_output(
    #                 target_audios, target_lengths, input_audios.shape[-1]
    #             )
    #         )
    #         target_audios, slice_start_idxs = rand_slice_audio_segments(
    #             target_audios_matched,
    #             target_matched_lengths,
    #             self.gen_segment_duration,
    #             self.vc_model.output_sample_frequency,
    #         )

    #     with torch.no_grad(), amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         xvector_output = self.xvector_model(
    #             input_audios,
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         speaker_feats = xvector_output.xvector

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         feats, feat_lengths = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=None,
    #             mode=FreeVCFwdMode.FEATS_ONLY,
    #         )
    #         vc_output = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.RECONS,
    #             slice_start_idxs=slice_start_idxs,
    #             slice_segment_length=int(
    #                 self.gen_segment_duration * self.vc_model.output_sample_frequency
    #             ),
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )

    #         y_real, _ = self.discrim_model(
    #             target_audios,
    #         )
    #         y_gen, _ = self.discrim_model(vc_output.gen_audio.detach())

    #     loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
    #         self.discrim_adv_loss(y_gen, y_real)
    #     )
    #     loss_discrim = loss_discrim / self.grad_acc_steps
    #     self.grad_scaler.scale(loss_discrim).backward()

    #     #######################################
    #     # 2. Generator Forward Reconstruction #
    #     #######################################
    #     self.discrim_model.set_train_mode(AudioDiscriminatorTrainMode.FROZEN)
    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         y_real, fmaps_real = self.discrim_model(target_audios)
    #         y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
    #         with torch.no_grad():
    #             mel_feats_real, mels_feats_real_lengths = self.audio_feats(
    #                 target_audios
    #             )

    #         mel_feats_gen, mel_feats_gen_lengths = self.audio_feats(
    #             vc_output.gen_audio.squeeze(1)
    #         )

    #     loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
    #     loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
    #     loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
    #     loss_kldiv = vc_output.kldiv_loss
    #     loss_gen_recons = (
    #         self.loss_gen_adv_weight * loss_gen_adv
    #         + self.loss_fm_weight * loss_fm
    #         + self.loss_mel_weight * loss_mel
    #         + self.loss_kl_weight * loss_kldiv
    #     ) / self.grad_acc_steps

    #     ###########################
    #     # 3. Generator Forward VC #
    #     ###########################
    #     with torch.no_grad():
    #         rand_perm = torch.randperm(len(speaker_feats), device=speaker_feats.device)
    #         speaker_feats = speaker_feats[rand_perm]

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         vc_output = self.vc_model(
    #             source_audios=None,
    #             source_audio_lengths=None,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.VC,
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )
    #         vc_audios_sliced = slice_segments(
    #             vc_output.gen_audio.squeeze(1),
    #             slice_start_idxs,
    #             int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
    #         ).unsqueeze(1)
    #         y_gen, fmaps_gen = self.discrim_model(vc_audios_sliced)
    #         xvector_output = self.xvector_model(
    #             vc_output.gen_audio.squeeze(1),
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         gen_speaker_feats = xvector_output.xvector

    #     loss_gen_adv_vc, losses_gen_adv_vc = self.gen_adv_loss(y_gen)
    #     loss_speaker_contrastive = self.speaker_contrastive_loss(
    #         gen_speaker_feats, speaker_feats
    #     )
    #     loss_gen_vc = (
    #         self.loss_vc_adv_weight * loss_gen_adv_vc
    #         + self.loss_speaker_contrastive_weight * loss_speaker_contrastive
    #     ) / self.grad_acc_steps

    #     if self.cur_step < self.vc_losses_warmup_start_step:
    #         vc_weight = 0.0
    #     elif (
    #         self.cur_step
    #         < self.vc_losses_warmup_steps + self.vc_losses_warmup_start_step
    #     ):
    #         # vc_weight = (
    #         #     1 - math.cos(math.pi * self.cur_step / self.vc_losses_warmup_steps)
    #         # ) / 2
    #         vc_weight = (
    #             self.cur_step - self.vc_losses_warmup_start_step
    #         ) / self.vc_losses_warmup_steps
    #     else:
    #         vc_weight = 1.0

    #     loss_gen_total = loss_gen_recons + vc_weight * loss_gen_vc
    #     self.grad_scaler.scale(loss_gen_total).backward()

    #     batch_metrics = ODict()
    #     batch_metrics["loss_discrim/total"] = loss_discrim.item() * self.grad_acc_steps
    #     batch_metrics["loss_gen/total_recons"] = (
    #         loss_gen_recons.item() * self.grad_acc_steps
    #     )
    #     batch_metrics["loss_gen/total_vc"] = loss_gen_vc.item() * self.grad_acc_steps
    #     batch_metrics["loss_gen/mel"] = loss_mel.item()
    #     batch_metrics["loss_gen/kldiv"] = loss_kldiv.item()
    #     batch_metrics["loss_gen/fm"] = loss_fm.item()
    #     batch_metrics["loss_gen/adv_recons"] = loss_gen_adv.item()
    #     batch_metrics["loss_gen/adv_vc"] = loss_gen_adv_vc.item()
    #     batch_metrics["loss_gen/speaker_contrastive"] = loss_speaker_contrastive.item()
    #     for i, loss in enumerate(losses_discrim_adv_gen):
    #         batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

    #     for i, loss in enumerate(losses_discrim_adv_real):
    #         batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv):
    #         batch_metrics[f"loss_gen_adv_recons/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv_vc):
    #         batch_metrics[f"loss_gen_adv_vc/{i}"] = loss

    #     return batch_metrics

    def train_forward_backward(self, batch_data):
        """Performs the forward and backward passes for both discriminator and generator.

        Handles discriminator training first with real/fake inputs, then updates
        the generator using adversarial and auxiliary losses.

        Returns:
            OrderedDict[str, float]: A dictionary of computed metrics.
        """
        self.speaker_contrastive_loss.update(self.cur_step)

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

        ###########################################
        # 1. Discriminator Forward Reconstruction #
        ###########################################
        self.discrim_model.set_train_mode(self.discrim_train_mode)
        with torch.no_grad(), amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            xvector_output = self.xvector_model(
                input_audios,
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            speaker_feats = xvector_output.xvector

        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            feats, feat_lengths = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=None,
                mode=FreeVCFwdMode.FEATS_ONLY,
            )
            vc_output = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.RECONS,
                slice_start_idxs=slice_start_idxs,
                slice_segment_length=int(
                    self.gen_segment_duration * self.vc_model.output_sample_frequency
                ),
                feats=feats,
                feat_lengths=feat_lengths,
            )

            y_real, _ = self.discrim_model(
                target_audios,
            )
            y_gen, _ = self.discrim_model(vc_output.gen_audio.detach())

        (
            loss_discrim_recons,
            losses_discrim_recons_adv_gen,
            losses_discrim_recons_adv_real,
        ) = self.discrim_adv_loss(y_gen, y_real)
        loss_discrim_recons = loss_discrim_recons / self.grad_acc_steps
        self.grad_scaler.scale(loss_discrim_recons).backward()

        #######################################
        # 2. Generator Forward Reconstruction #
        #######################################
        self.discrim_model.set_train_mode(AudioDiscriminatorTrainMode.FROZEN)
        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            y_real, fmaps_real = self.discrim_model(target_audios)
            y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
            with torch.no_grad():
                mel_feats_real, mels_feats_real_lengths = self.audio_feats(
                    target_audios
                )

            mel_feats_gen, mel_feats_gen_lengths = self.audio_feats(
                vc_output.gen_audio.squeeze(1)
            )

        loss_mrfb_log_mag, loss_mrfb_conv = self.mrfb_loss(
            vc_output.gen_audio.squeeze(1), target_audios
        )
        loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
        loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
        loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
        loss_kldiv = vc_output.kldiv_loss
        loss_gen_recons = (
            self.loss_gen_adv_weight * loss_gen_adv
            + self.loss_fm_weight * loss_fm
            + self.loss_mel_weight * loss_mel
            + self.loss_mrfb_log_mag_weight * loss_mrfb_log_mag
            + self.loss_mrfb_conv_weight * loss_mrfb_conv
            + self.loss_kl_weight * loss_kldiv
        ) / self.grad_acc_steps

        ###########################
        # 3. Generator Forward VC #
        ###########################
        with torch.no_grad():
            rand_perm = torch.randperm(len(speaker_feats), device=speaker_feats.device)
            speaker_feats = speaker_feats[rand_perm]

        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            vc_output = self.vc_model(
                source_audios=None,
                source_audio_lengths=None,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.VC,
                feats=feats,
                feat_lengths=feat_lengths,
            )
            vc_audios_sliced = slice_segments(
                vc_output.gen_audio.squeeze(1),
                slice_start_idxs,
                int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
            ).unsqueeze(1)
            y_gen, fmaps_gen = self.discrim_model(vc_audios_sliced)
            xvector_output = self.xvector_model(
                vc_output.gen_audio.squeeze(1),
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            gen_speaker_feats = xvector_output.xvector

        loss_gen_adv_vc, losses_gen_adv_vc = self.gen_adv_loss(y_gen)
        loss_speaker_contrastive = self.speaker_contrastive_loss(
            gen_speaker_feats, speaker_feats
        )
        loss_gen_vc = (
            self.loss_vc_adv_weight * loss_gen_adv_vc
            + self.loss_speaker_contrastive_weight * loss_speaker_contrastive
        ) / self.grad_acc_steps

        if self.cur_step < self.vc_losses_warmup_start_step:
            vc_weight = 0.0
        elif (
            self.cur_step
            < self.vc_losses_warmup_steps + self.vc_losses_warmup_start_step
        ):
            # vc_weight = (
            #     1 - math.cos(math.pi * self.cur_step / self.vc_losses_warmup_steps)
            # ) / 2
            vc_weight = (
                self.cur_step - self.vc_losses_warmup_start_step
            ) / self.vc_losses_warmup_steps
        else:
            vc_weight = 1.0

        loss_gen_total = loss_gen_recons + vc_weight * loss_gen_vc
        self.grad_scaler.scale(loss_gen_total).backward()

        ###########################################
        # 4. Discriminator Forward VC             #
        ###########################################
        self.discrim_model.set_train_mode(self.discrim_train_mode)
        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            y_real, _ = self.discrim_model(
                target_audios,
            )
            y_gen, _ = self.discrim_model(vc_audios_sliced.detach())

        loss_discrim_vc, losses_discrim_vc_adv_gen, losses_discrim_vc_adv_real = (
            self.discrim_adv_loss(y_gen, y_real)
        )
        loss_discrim_vc = loss_discrim_vc / self.grad_acc_steps
        loss_discrim_vc_total = vc_weight * loss_discrim_vc
        if vc_weight > 0.0:
            self.grad_scaler.scale(loss_discrim_vc_total).backward()

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total_recons"] = (
            loss_discrim_recons.item() * self.grad_acc_steps
        )
        batch_metrics["loss_discrim/total_vc"] = (
            loss_discrim_vc.item() * self.grad_acc_steps
        )
        batch_metrics["loss_gen/total_recons"] = (
            loss_gen_recons.item() * self.grad_acc_steps
        )
        batch_metrics["loss_gen/total_vc"] = loss_gen_vc.item() * self.grad_acc_steps
        batch_metrics["loss_gen/mel"] = loss_mel.item()
        batch_metrics["loss_gen/mrfb_log_mag"] = loss_mrfb_log_mag.item()
        batch_metrics["loss_gen/mrfb_conv"] = loss_mrfb_conv.item()
        batch_metrics["loss_gen/kldiv"] = loss_kldiv.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv_recons"] = loss_gen_adv.item()
        batch_metrics["loss_gen/adv_vc"] = loss_gen_adv_vc.item()
        batch_metrics["loss_gen/speaker_contrastive"] = loss_speaker_contrastive.item()
        for i, loss in enumerate(losses_discrim_recons_adv_gen):
            batch_metrics[f"loss_discrim_recons_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_recons_adv_real):
            batch_metrics[f"loss_discrim_recons_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_discrim_vc_adv_gen):
            batch_metrics[f"loss_discrim_vc_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_vc_adv_real):
            batch_metrics[f"loss_discrim_vc_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv_recons/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv_vc):
            batch_metrics[f"loss_gen_adv_vc/{i}"] = loss

        return batch_metrics

    # def train_forward_backward_2(self, batch_data):
    #     """Performs the forward and backward passes for both discriminator and generator.

    #     Handles discriminator training first with real/fake inputs, then updates
    #     the generator using adversarial and auxiliary losses.

    #     Returns:
    #         OrderedDict[str, float]: A dictionary of computed metrics.
    #     """
    #     self.speaker_contrastive_loss.update(self.cur_step)

    #     input_audios, input_lengths = (
    #         batch_data[f"source_audios"],
    #         batch_data[f"source_audio_lengths"],
    #     )

    #     target_audios, target_lengths = (
    #         batch_data[f"target_audios"],
    #         batch_data[f"target_audio_lengths"],
    #     )
    #     # print(batch_data, flush=True)
    #     with torch.no_grad():
    #         target_audios_matched, target_matched_lengths = (
    #             self.vc_model.get_target_matching_output(
    #                 target_audios, target_lengths, input_audios.shape[-1]
    #             )
    #         )
    #         target_audios, slice_start_idxs = rand_slice_audio_segments(
    #             target_audios_matched,
    #             target_matched_lengths,
    #             self.gen_segment_duration,
    #             self.vc_model.output_sample_frequency,
    #         )

    #     ###########################################
    #     # 1. Discriminator Forward Reconstruction #
    #     ###########################################
    #     self.discrim_model.set_train_mode(self.discrim_train_mode)
    #     with torch.no_grad(), amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         xvector_output = self.xvector_model(
    #             input_audios,
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         speaker_feats = xvector_output.xvector

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         feats, feat_lengths = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=None,
    #             mode=FreeVCFwdMode.FEATS_ONLY,
    #         )
    #         vc_output = self.vc_model(
    #             source_audios=input_audios,
    #             source_audio_lengths=input_lengths,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.RECONS,
    #             slice_start_idxs=slice_start_idxs,
    #             slice_segment_length=int(
    #                 self.gen_segment_duration * self.vc_model.output_sample_frequency
    #             ),
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )

    #         y_real, _ = self.discrim_model(
    #             target_audios,
    #         )
    #         y_gen, _ = self.discrim_model(vc_output.gen_audio.detach())

    #     (
    #         loss_discrim_recons,
    #         losses_discrim_recons_adv_gen,
    #         losses_discrim_recons_adv_real,
    #     ) = self.discrim_adv_loss(y_gen, y_real)
    #     loss_discrim_recons = loss_discrim_recons / self.grad_acc_steps
    #     self.grad_scaler.scale(loss_discrim_recons).backward()

    #     #######################################
    #     # 2. Generator Forward Reconstruction #
    #     #######################################
    #     self.discrim_model.set_train_mode(AudioDiscriminatorTrainMode.FROZEN)
    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         y_real, fmaps_real = self.discrim_model(target_audios)
    #         y_gen, fmaps_gen = self.discrim_model(vc_output.gen_audio)
    #         with torch.no_grad():
    #             mel_feats_real, mels_feats_real_lengths = self.audio_feats(
    #                 target_audios
    #             )

    #         mel_feats_gen, mel_feats_gen_lengths = self.audio_feats(
    #             vc_output.gen_audio.squeeze(1)
    #         )

    #     loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
    #     loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
    #     loss_mel = self.l1_loss(mel_feats_gen, mel_feats_real)
    #     loss_kldiv = vc_output.kldiv_loss
    #     loss_gen_recons = (
    #         self.loss_gen_adv_weight * loss_gen_adv
    #         + self.loss_fm_weight * loss_fm
    #         + self.loss_mel_weight * loss_mel
    #         + self.loss_kl_weight * loss_kldiv
    #     ) / self.grad_acc_steps

    #     ###########################
    #     # 3. Generator Forward VC #
    #     ###########################
    #     with torch.no_grad():
    #         rand_perm = torch.randperm(len(speaker_feats), device=speaker_feats.device)
    #         speaker_feats = speaker_feats[rand_perm]

    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         vc_output = self.vc_model(
    #             source_audios=None,
    #             source_audio_lengths=None,
    #             speaker_feats=speaker_feats,
    #             mode=FreeVCFwdMode.VC,
    #             feats=feats,
    #             feat_lengths=feat_lengths,
    #         )
    #         vc_audios_sliced = slice_segments(
    #             vc_output.gen_audio.squeeze(1),
    #             slice_start_idxs,
    #             int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
    #         ).unsqueeze(1)
    #         y_gen, fmaps_gen = self.discrim_model(vc_audios_sliced)
    #         xvector_output = self.xvector_model(
    #             vc_output.gen_audio.squeeze(1),
    #             input_lengths,
    #             return_classif_layers=[0],
    #             return_logits=False,
    #         )
    #         gen_speaker_feats = xvector_output.xvector

    #     loss_gen_adv_vc, losses_gen_adv_vc = self.gen_adv_loss(y_gen)
    #     loss_speaker_contrastive = self.speaker_contrastive_loss(
    #         gen_speaker_feats, speaker_feats
    #     )
    #     loss_gen_vc = (
    #         self.loss_vc_adv_weight * loss_gen_adv_vc
    #         + self.loss_speaker_contrastive_weight * loss_speaker_contrastive
    #     ) / self.grad_acc_steps

    #     if self.cur_step < self.vc_losses_warmup_start_step:
    #         vc_weight = 0.0
    #     elif (
    #         self.cur_step
    #         < self.vc_losses_warmup_steps + self.vc_losses_warmup_start_step
    #     ):
    #         # vc_weight = (
    #         #     1 - math.cos(math.pi * self.cur_step / self.vc_losses_warmup_steps)
    #         # ) / 2
    #         vc_weight = (
    #             self.cur_step - self.vc_losses_warmup_start_step
    #         ) / self.vc_losses_warmup_steps
    #     else:
    #         vc_weight = 1.0

    #     loss_gen_total = loss_gen_recons + vc_weight * loss_gen_vc
    #     self.grad_scaler.scale(loss_gen_total).backward()

    #     ###########################################
    #     # 4. Discriminator Forward VC             #
    #     ###########################################
    #     self.discrim_model.set_train_mode(self.discrim_train_mode)
    #     with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
    #         y_real, _ = self.discrim_model(
    #             target_audios,
    #         )
    #         y_gen, _ = self.discrim_model(vc_audios_sliced.detach())

    #     loss_discrim_vc, losses_discrim_vc_adv_gen, losses_discrim_vc_adv_real = (
    #         self.discrim_adv_loss(y_gen, y_real)
    #     )
    #     loss_discrim_vc = loss_discrim_vc / self.grad_acc_steps
    #     loss_discrim_vc_total = vc_weight * loss_discrim_vc
    #     if vc_weight > 0.0:
    #         self.grad_scaler.scale(loss_discrim_vc_total).backward()

    #     batch_metrics = ODict()
    #     batch_metrics["loss_discrim/total_recons"] = (
    #         loss_discrim_recons.item() * self.grad_acc_steps
    #     )
    #     batch_metrics["loss_discrim/total_vc"] = (
    #         loss_discrim_vc.item() * self.grad_acc_steps
    #     )
    #     batch_metrics["loss_gen/total_recons"] = (
    #         loss_gen_recons.item() * self.grad_acc_steps
    #     )
    #     batch_metrics["loss_gen/total_vc"] = loss_gen_vc.item() * self.grad_acc_steps
    #     batch_metrics["loss_gen/mel"] = loss_mel.item()
    #     batch_metrics["loss_gen/kldiv"] = loss_kldiv.item()
    #     batch_metrics["loss_gen/fm"] = loss_fm.item()
    #     batch_metrics["loss_gen/adv_recons"] = loss_gen_adv.item()
    #     batch_metrics["loss_gen/adv_vc"] = loss_gen_adv_vc.item()
    #     batch_metrics["loss_gen/speaker_contrastive"] = loss_speaker_contrastive.item()
    #     for i, loss in enumerate(losses_discrim_recons_adv_gen):
    #         batch_metrics[f"loss_discrim_recons_adv_gen/{i}"] = loss

    #     for i, loss in enumerate(losses_discrim_recons_adv_real):
    #         batch_metrics[f"loss_discrim_recons_adv_real/{i}"] = loss

    #     for i, loss in enumerate(losses_discrim_vc_adv_gen):
    #         batch_metrics[f"loss_discrim_vc_adv_gen/{i}"] = loss

    #     for i, loss in enumerate(losses_discrim_vc_adv_real):
    #         batch_metrics[f"loss_discrim_vc_adv_real/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv):
    #         batch_metrics[f"loss_gen_adv_recons/{i}"] = loss

    #     for i, loss in enumerate(losses_gen_adv_vc):
    #         batch_metrics[f"loss_gen_adv_vc/{i}"] = loss

    #     return batch_metrics

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

        #######################################
        # 1. Inference Forward Reconstruction #
        #######################################
        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            xvector_output = self.xvector_model(
                input_audios,
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            speaker_feats = xvector_output.xvector
            feats, feat_lengths = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=None,
                mode=FreeVCFwdMode.FEATS_ONLY,
            )
            vc_output = self.vc_model(
                source_audios=input_audios,
                source_audio_lengths=input_lengths,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.VC,
                feats=feats,
                feat_lengths=feat_lengths,
            )
            # if (
            #     vc_output.z.shape[1]
            #     != self.vc_model.hf_feats.max_out_length(input_audios.shape[-1])
            #     or target_audios.shape[1] != vc_output.gen_audio.shape[-1]
            # ):
            #     print(
            #         "sliced",
            #         input_audios.shape,
            #         input_lengths,
            #         target_audios.shape,
            #         target_lengths,
            #         target_audios_sliced.shape,
            #         vc_output.gen_audio.shape,
            #         vc_output.z.shape,
            #         slice_start_idxs,
            #         int(
            #             self.gen_segment_duration
            #             * self.vc_model.output_sample_frequency
            #         ),
            #         self.vc_model.hf_feats.max_out_length(input_audios.shape[-1]),
            #         flush=True,
            #     )
            gen_audios_sliced = slice_segments(
                vc_output.gen_audio.squeeze(1),
                slice_start_idxs,
                int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
            )
            y_real, fmaps_real = self.discrim_model(target_audios_sliced)
            y_gen, fmaps_gen = self.discrim_model(gen_audios_sliced)
            mel_feats_real, _ = self.audio_feats(target_audios)
            mel_feats_gen, _ = self.audio_feats(vc_output.gen_audio.squeeze(1))
            loss_mrfb_log_mag, loss_mrfb_conv = self.mrfb_loss(
                vc_output.gen_audio.squeeze(1), target_audios
            )

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
            + self.loss_mrfb_log_mag_weight * loss_mrfb_log_mag
            + self.loss_mrfb_conv_weight * loss_mrfb_conv
        )

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total_recons"] = loss_discrim.item()
        batch_metrics["loss_gen/total_recons"] = loss_gen.item()
        batch_metrics["loss_gen/mel"] = loss_mel.item()
        batch_metrics["loss_gen/mrfb_log_mag"] = loss_mrfb_log_mag.item()
        batch_metrics["loss_gen/mrfb_conv"] = loss_mrfb_conv.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv_recons"] = loss_adv_gen.item()
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_recons_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_recons_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv_recons/{i}"] = loss

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
                f"audios_generated_recons/{_id}",
                vc_output.gen_audio[i, 0],
                sample_freq=self.vc_model.output_sample_frequency,
            )
            self.loggers.log_spectrogram(
                f"log_mel_fbanks_target/{_id}", mel_feats_real[i]
            )
            self.loggers.log_spectrogram(
                f"log_mel_fbanks_generated_recons/{_id}", mel_feats_gen[i]
            )

        ###########################
        # 2. Inference Forward VC #
        ###########################
        # shift_speaker_feats = torch.zeros_like(speaker_feats)
        # shift_speaker_feats[:-1] = speaker_feats[1:]
        # shift_speaker_feats[-1] = speaker_feats[0]
        # speaker_feats = shift_speaker_feats
        speaker_feats = torch.flip(
            speaker_feats, dims=[0]
        )  # reverse the order of speaker feats
        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            vc_output = self.vc_model(
                source_audios=None,
                source_audio_lengths=None,
                speaker_feats=speaker_feats,
                mode=FreeVCFwdMode.VC,
                feats=feats,
                feat_lengths=feat_lengths,
            )
            vc_audios_sliced = slice_segments(
                vc_output.gen_audio.squeeze(1),
                slice_start_idxs,
                int(self.gen_segment_duration * self.vc_model.output_sample_frequency),
            ).unsqueeze(1)
            y_gen, _ = self.discrim_model(vc_audios_sliced)
            xvector_output = self.xvector_model(
                vc_output.gen_audio.squeeze(1),
                input_lengths,
                return_classif_layers=[0],
                return_logits=False,
            )
            gen_speaker_feats = xvector_output.xvector
            mel_feats_gen, _ = self.audio_feats(vc_output.gen_audio.squeeze(1))

        loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
            self.discrim_adv_loss(y_gen, y_real)
        )
        loss_gen_adv_vc, losses_gen_adv_vc = self.gen_adv_loss(y_gen)
        loss_speaker_contrastive = self.speaker_contrastive_loss(
            gen_speaker_feats, speaker_feats
        )
        loss_gen_vc = (
            self.loss_vc_adv_weight * loss_gen_adv_vc
            + self.loss_speaker_contrastive_weight * loss_speaker_contrastive
        )
        batch_metrics["loss_discrim/total_vc"] = loss_discrim.item()
        batch_metrics["loss_gen/total_vc"] = loss_gen_vc.item()
        batch_metrics["loss_gen/adv_vc"] = loss_adv_gen.item()
        batch_metrics["loss_gen/speaker_contrastive"] = loss_speaker_contrastive.item()
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_vc_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_vc_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv_vc):
            batch_metrics[f"loss_gen_adv_vc/{i}"] = loss

        for i in range(num_log_samples):
            _id = batch_data["id"][i]
            self.loggers.log_audio(
                f"audios_generated_vc/{_id}",
                vc_output.gen_audio[i, 0],
                sample_freq=self.vc_model.output_sample_frequency,
            )
            self.loggers.log_audio(
                f"audios_ref_vc/{_id}",
                target_audios[-1 - i],
                sample_freq=self.vc_model.output_sample_frequency,
            )
            self.loggers.log_spectrogram(
                f"log_mel_fbanks_generated_vc/{_id}", mel_feats_gen[i]
            )

        self.cur_val_log_samples += num_log_samples

        return batch_size, batch_metrics

    # def update_models(self):
    #     """Steps optimizers and schedulers for both generator and discriminator.

    #     Also clips gradients and returns logs for gradient norms.

    #     Returns:
    #         Dict[str, float]: Gradient norm logs.
    #     """

    #     for sch in [self.vc_lr_scheduler, self.discrim_lr_scheduler]:
    #         if sch is not None and not self.in_swa:
    #             sch.on_opt_step()

    #     for sch in [self.vc_wd_scheduler, self.discrim_wd_scheduler]:
    #         if sch is not None:
    #             sch.on_opt_step()

    #     vc_grad_norm = self._update_model_by_optim(
    #         self.vc_model,
    #         self.vc_optimizer,
    #         self.grad_clip,
    #         self.grad_clip_norm,
    #         self.use_amp,
    #         self.grad_scaler,
    #     )
    #     discrim_grad_norm = self._update_model_by_optim(
    #         self.discrim_model,
    #         self.discrim_optimizer,
    #         self.grad_clip,
    #         self.grad_clip_norm,
    #         self.use_amp,
    #         self.grad_scaler,
    #     )
    #     self.grad_scaler.update()

    #     logs = {"grad_norm/vc": vc_grad_norm, "grad_norm/discrim": discrim_grad_norm}
    #     return logs

    # def save_checkpoint(self, logs=None):
    #     """Saves current training state to disk, including both models and optionally SWA.

    #     Args:
    #         logs (Optional[Dict[str, Any]]): Logging metrics to include in the checkpoint.
    #     """
    #     if self.ddp and (
    #         self.ddp_type == DDPType.OSS_DDP or self.ddp_type == DDPType.OSS_SHARDED_DDP
    #     ):
    #         # Not sure what this does, just copying from the example in
    #         # https://github.com/facebookresearch/fairscale/blob/master/benchmarks/oss.py
    #         # Check the checkpointing in the case of the OSS optimizer
    #         # Memory usage could spill over from there
    #         # optimizer = cast(OSS, optimizer)
    #         self.vc_optimizer.consolidate_state_dict()
    #         self.discrim_optimizer.consolidate_state_dict()

    #     if self.rank != 0:
    #         return

    #     checkpoint = self.model_checkpoint(
    #         self.vc_model,
    #         self.vc_optimizer,
    #         self.vc_lr_scheduler,
    #         self.vc_wd_scheduler,
    #         self.swa_vc_model,
    #         self.swa_vc_scheduler,
    #         logs=logs,
    #     )

    #     self.save_model_checkpoint_to_file("vc_model", checkpoint)

    #     checkpoint = self.model_checkpoint(
    #         self.discrim_model,
    #         self.discrim_optimizer,
    #         self.discrim_lr_scheduler,
    #         self.discrim_wd_scheduler,
    #         logs=logs,
    #     )

    #     self.save_model_checkpoint_to_file("discrim_model", checkpoint)

    # def save_swa_model(self, logs=None):
    #     """Saves the final SWA-averaged generator model to disk.

    #     Args:
    #         logs (Optional[Dict[str, Any]]): Logging metrics to include in the checkpoint.
    #     """
    #     if self.rank != 0:
    #         return

    #     checkpoint = self.checkpoint(
    #         self.vc_model,
    #         self.vc_optimizer,
    #         self.vc_lr_scheduler,
    #         self.vc_wd_scheduler,
    #         self.swa_vc_model,
    #         self.swa_vc_scheduler,
    #         logs=logs,
    #     )
    #     checkpoint["model_state_dict"] = checkpoint["swa_model_state_dict"]
    #     del checkpoint["swa_model_state_dict"]
    #     file_path = "%s/swa_vc_model_ep%04d_%010d.pth" % (
    #         self.exp_path,
    #         self.cur_epoch,
    #         self.cur_step,
    #     )
    #     torch.save(checkpoint, file_path)

    # def load_checkpoint(self, epoch, step):
    #     """Loads training state from checkpoint files for both generator and discriminator.

    #     Args:
    #         epoch (int): Epoch number of checkpoint.
    #         step (int): Step number of checkpoint.

    #     Returns:
    #         Optional[Dict[str, Any]]: Logs saved with the checkpoint, if any.
    #     """
    #     checkpoint = self.load_model_checkpoint_from_file("vc_model", epoch, step)
    #     logs = self._load_vars_from_checkpoint(checkpoint)
    #     self._load_model_state_dicts_from_checkpoint(
    #         checkpoint,
    #         self.vc_model,
    #         self.vc_optimizer,
    #         self.vc_lr_scheduler,
    #         self.vc_wd_scheduler,
    #         self.swa_vc_model,
    #         self.swa_vc_scheduler,
    #     )
    #     checkpoint = self.load_model_checkpoint_from_file("discrim_model", epoch, step)
    #     self._load_model_state_dicts_from_checkpoint(
    #         checkpoint,
    #         self.discrim_model,
    #         self.discrim_optimizer,
    #         self.discrim_lr_scheduler,
    #         self.discrim_wd_scheduler,
    #     )
    #     return logs

    @staticmethod
    def filter_args(**kwargs):
        """
        Filters the provided keyword arguments to retain only those valid for the FreeVCTrainer constructor.

        Returns:
            dict: A dictionary of filtered arguments applicable to VIAnonymizerTrainer.
        """
        args = filter_func_args(VIAnonymizerTrainer.__init__, kwargs)
        return args

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

        FreeVCTrainer.add_io_keys_args(parser)
        parser.add_argument(
            "--speaker-key",
            default="speaker",
            help="Key used to access speaker information in the batch dictionary.",
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_loss_weights_args(parser, prefix=None):
        """
        Adds command-line arguments to configure loss weights for the generator.

        Args:
            parser (ArgumentParser): Argument parser to which arguments are added.
            prefix (str, optional): Optional namespace prefix.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        FreeVCTrainer.add_loss_weights_args(parser, skip={"loss_mel_weight"})
        parser.add_argument(
            "--loss-mel-weight",
            default=0.0,
            type=float,
            help="Weight for the mel-spectrogram L1 loss.",
        )
        parser.add_argument(
            "--loss_mrfb_log_mag_weight",
            default=4.5,
            type=float,
            help="Weight for the multi-resolution filter bank log-magnitude loss.",
        )
        parser.add_argument(
            "--loss_mrfb_conv_weight",
            default=1.0,
            type=float,
            help="Weight for the multi-resolution filter bank complex convolution loss.",
        )
        parser.add_argument(
            "--loss-vc-adv-weight",
            default=1.0,
            type=float,
            help="Weight for the adversarial loss in the voice conversion component.",
        )
        parser.add_argument(
            "--loss-speaker-contrastive-weight",
            default=1.0,
            type=float,
            help="Weight for the speaker contrastive loss.",
        )
        parser.add_argument(
            "--vc-losses-warmup-steps",
            default=50000,
            type=int,
            help="Number of steps to warm up the voice conversion losses.",
        )
        parser.add_argument(
            "--vc-losses-warmup-start-step",
            default=50000,
            type=int,
            help="Step to start warming up the voice conversion losses.",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        """
        Adds all VIAnonymizerTrainer-related arguments to the parser, including trainer, optimizer, I/O, and loss configuration.

        Args:
            parser (ArgumentParser): Argument parser to which arguments are added.
            prefix (str, optional): Optional prefix to namespace all arguments.
            skip (set): Set of argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        TorchTrainerBase.add_class_args(parser)
        MultiResolutionFilterBankLoss.add_class_args(parser, prefix="mrfb_loss")
        VIAnonymizerTrainer.add_optim_args(parser)
        VIAnonymizerTrainer.add_io_keys_args(parser)
        VIAnonymizerTrainer.add_train_modes_args(parser)
        VIAnonymizerTrainer.add_loss_weights_args(parser)
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
        ContrastiveLoss.add_class_args(parser, prefix="speaker_contrastive_loss")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

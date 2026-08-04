"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from collections import OrderedDict as ODict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.amp as amp
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import PathLike, filter_func_args
from ..hyper_torch_model import HyperTorchModel
from ..loggers import LoggerList
from ..losses import (
    AudioDiscriminatorAdvLoss,
    AudioGeneratorAdvLoss,
    FeatureMatchingLoss,
    MultiResolutionFilterBankLoss,
)
from ..lr_schedulers import LRScheduler as LRS
from ..lr_schedulers import LRSchedulerFactory as LRSF
from ..models.audio_discriminator.audio_multi_discriminator import (
    AudioDiscriminatorTrainMode,
)
from ..models.dac.dac import DACTrainMode
from ..optim import OptimizerFactory as OF
from ..utils.misc import rand_slice_audio_segments, slice_segments
from ..wd_schedulers import WDScheduler as WDS
from ..wd_schedulers import WDSchedulerFactory as WDSF
from .torch_trainer_base import AMPDType, DDPType, FSDPMPDType, TorchTrainerBase


class DACTrainer(TorchTrainerBase):
    """Trainer for DAC models.

    This trainer handles the training of both the voice conversion (generator) and
    discriminator models used in adversarial training setups. It supports mixed precision,
    SWA, DDP, and advanced logging through TensorBoard and W&B.

    Attributes:
        dac_model: Generator model for voice conversion.
        discrim_model: Discriminator model for adversarial training.
        dac_optim: Optimizer for the generator.
        discrim_optim: Optimizer for the discriminator.
        dac_lrsched: Learning rate scheduler for the generator.
        discrim_lrsched: Learning rate scheduler for the discriminator.
        dac_wdsched: Weight decay scheduler for the generator.
        discrim_wdsched: Weight decay scheduler for the discriminator.
        dac_train_mode: Training mode for the generator.
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
        ddp_type (DDPType): Type of DDP implementation to use (standard DDP or torch FSDP2).
        fsdp_reshard_after_forward (bool|int|None): FSDP2 reshard policy after forward.
        fsdp_mp_param_dtype (FSDPMPDType|None): FSDP2 mixed-precision parameter dtype.
        fsdp_mp_reduce_dtype (FSDPMPDType|None): FSDP2 mixed-precision reduction dtype.
        fsdp_mp_output_dtype (FSDPMPDType|None): FSDP2 mixed-precision output dtype.
        fsdp_cpu_offload (bool): Whether to offload parameters/gradients to CPU in FSDP2.
        use_amp (bool): Enables mixed-precision training using AMP (Automatic Mixed Precision).
        amp_dtype (AMPDType): Data type for AMP (float16 or bfloat16).
        bf16_grad_scaler (bool): Enable GradScaler when using bfloat16.
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
        loss_mrfb_log_mag_weight: Weight for the mel spectrogram loss.
        loss_mrfb_conv_weight: Weight for the mel spectrogram loss.
        loss_kl_weight: Weight for the KL divergence loss.
        loss_gen_adv_weight: Weight for adversarial generator loss.
        loss_fm_weight: Weight for feature matching loss.
        gen_adv_losses_start_steps: Steps to keep adversarial and feature matching losses at 0.
        gen_adv_losses_warmup_steps: Steps to ramp adversarial and feature matching losses to full weight.
        num_val_log_samples: Max number of samples to log during validation.
        context_trim_fraction: Fraction of receptive-field context removed from
            both ends before computing losses (0.0 keeps current behavior).
    """

    checkpoint_model_names = ("dac_model", "discrim_model")

    def __init__(
        self,
        dac_model: HyperTorchModel,
        discrim_model: HyperTorchModel,
        mrfb_loss: Union[MultiResolutionFilterBankLoss, Dict[str, Any]],
        dac_optim: torch.optim.Optimizer,
        discrim_optim: torch.optim.Optimizer,
        dac_lrsched: Optional[LRS] = None,
        discrim_lrsched: Optional[LRS] = None,
        dac_wdsched: Optional[WDS] = None,
        discrim_wdsched: Optional[WDS] = None,
        dac_train_mode: DACTrainMode = DACTrainMode.FULL,
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
        fsdp_reshard_after_forward: Optional[Union[bool, int]] = None,
        fsdp_mp_param_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_reduce_dtype: Optional[FSDPMPDType] = None,
        fsdp_mp_output_dtype: Optional[FSDPMPDType] = None,
        fsdp_cpu_offload: bool = False,
        use_amp: bool = False,
        amp_dtype: AMPDType = AMPDType.FLOAT16,
        bf16_grad_scaler: bool = False,
        log_interval: int = 1000,
        log_gpu_usage: bool = False,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb: Dict[str, str] = {},
        grad_clip: float = 0,
        discrim_grad_clip: float = 0,
        grad_clip_norm: Union[str, int] = 2,
        swa_start: int = 0,
        swa_lr: float = 1e-3,
        swa_anneal_steps: int = 50000,
        input_audio_key: str = "audio",
        target_audio_key: str = "audio",
        loss_mrfb_log_mag_weight: float = 15.0,
        loss_mrfb_conv_weight: float = 15.0,
        loss_gen_adv_weight: float = 1.0,
        loss_fm_weight: float = 1.0,
        loss_codebook_weight: float = 1.0,
        loss_commitment_weight: float = 0.25,
        loss_orthogonality_weight: float = 0.0,
        loss_diversity_weight: float = 0.0,
        gen_adv_losses_warmup_steps: int = 0,
        gen_adv_losses_start_steps: int = 0,
        context_trim_fraction: float = 0.0,
        # gen_segment_duration: float = 0.64,
        num_val_log_samples: int = 10,
    ):
        """Initializes the DAC trainer.

        Args:
            dac_model: Generator model to optimize.
            discrim_model: Discriminator model to optimize.
            mrfb_loss: Multi-resolution filter bank loss or its config.
            dac_optim: Optimizer for the generator.
            discrim_optim: Optimizer for the discriminator.
            dac_lrsched: Optional generator learning-rate scheduler.
            discrim_lrsched: Optional discriminator learning-rate scheduler.
            dac_wdsched: Optional generator weight-decay scheduler.
            discrim_wdsched: Optional discriminator weight-decay scheduler.
            dac_train_mode: Training mode for the generator.
            discrim_train_mode: Training mode for the discriminator.
            exp_path: Output directory for checkpoints and logs.
            num_epochs: Total number of training epochs.
            cur_epoch: Initial epoch index.
            max_steps: Maximum number of optimizer steps.
            cur_step: Initial step index.
            grad_acc_steps: Gradient accumulation steps.
            eff_batch_size: Effective batch size across workers.
            val_steps: Validation interval in steps.
            save_steps: Checkpoint interval in steps.
            device: Training device.
            loggers: Logger collection.
            ddp: Whether distributed training is enabled.
            ddp_type: Distributed-training backend type.
            fsdp_reshard_after_forward: FSDP reshard policy.
            fsdp_mp_param_dtype: FSDP parameter mixed-precision dtype.
            fsdp_mp_reduce_dtype: FSDP reduction mixed-precision dtype.
            fsdp_mp_output_dtype: FSDP output mixed-precision dtype.
            fsdp_cpu_offload: Whether to offload FSDP parameters to CPU.
            use_amp: Whether to enable AMP.
            amp_dtype: AMP floating-point dtype.
            bf16_grad_scaler: Whether to use GradScaler with bfloat16.
            log_interval: Logging interval in steps.
            log_gpu_usage: Whether to log GPU memory usage.
            use_tensorboard: Whether to enable TensorBoard logging.
            use_wandb: Whether to enable Weights & Biases logging.
            wandb: Weights & Biases configuration.
            grad_clip: Generator gradient clipping norm.
            discrim_grad_clip: Discriminator gradient clipping norm.
            grad_clip_norm: Gradient norm type.
            swa_start: Step to start SWA.
            swa_lr: SWA learning rate.
            swa_anneal_steps: SWA annealing steps.
            input_audio_key: Batch key for input audio.
            target_audio_key: Batch key for target audio.
            loss_mrfb_log_mag_weight: Weight for the log-magnitude MRFB loss.
            loss_mrfb_conv_weight: Weight for the convolutional MRFB loss.
            loss_gen_adv_weight: Weight for adversarial generator loss.
            loss_fm_weight: Weight for feature-matching loss.
            loss_codebook_weight: Weight for codebook loss.
            loss_commitment_weight: Weight for commitment loss.
            loss_orthogonality_weight: Weight for orthogonality loss.
            loss_diversity_weight: Weight for diversity loss.
            gen_adv_losses_warmup_steps: Warmup steps for adversarial losses.
            gen_adv_losses_start_steps: Delay before adversarial losses turn on.
            context_trim_fraction: Fraction of context to trim from both ends.
            num_val_log_samples: Maximum number of validation samples to log.
        """

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)

        self.dac_model = dac_model
        self.discrim_model = discrim_model
        self.dac_optim = dac_optim
        self.dac_lrsched = dac_lrsched
        self.dac_wdsched = dac_wdsched
        self.discrim_optim = discrim_optim
        self.discrim_lrsched = discrim_lrsched
        self.discrim_wdsched = discrim_wdsched
        self.dac_train_mode = dac_train_mode
        self.discrim_train_mode = discrim_train_mode
        self.input_audio_key = input_audio_key
        self.target_audio_key = target_audio_key
        self.loss_mrfb_log_mag_weight = loss_mrfb_log_mag_weight
        self.loss_mrfb_conv_weight = loss_mrfb_conv_weight
        self.loss_gen_adv_weight = loss_gen_adv_weight
        self.loss_fm_weight = loss_fm_weight
        self.loss_codebook_weight = loss_codebook_weight
        self.loss_commitment_weight = loss_commitment_weight
        self.loss_orthogonality_weight = loss_orthogonality_weight
        self.loss_diversity_weight = loss_diversity_weight
        # self.gen_segment_duration = gen_segment_duration
        self.num_val_log_samples = num_val_log_samples
        self.cur_val_log_samples = 0
        self.discrim_grad_clip = discrim_grad_clip
        self.gen_adv_losses_warmup_steps = gen_adv_losses_warmup_steps
        self.gen_adv_losses_start_steps = gen_adv_losses_start_steps
        self.context_trim_fraction = context_trim_fraction
        if self.context_trim_fraction > 0.0:
            lc, rc = self.dac_model.in_context()
            self._left_contest_trim = int(lc * self.context_trim_fraction)
            self._right_contest_trim = int(rc * self.context_trim_fraction)
            logging.info(
                f"Trimming {self._left_contest_trim} samples from left and "
                f"{self._right_contest_trim} samples from right for context."
            )
        else:
            self._left_contest_trim = 0
            self._right_contest_trim = 0

        self.set_train_mode()
        self.prepare_models_for_training()
        if isinstance(mrfb_loss, dict):
            self.mrfb_loss = MultiResolutionFilterBankLoss(**mrfb_loss).to(self.device)
        else:
            self.mrfb_loss = mrfb_loss.to(self.device)

        if self.rank == 0:
            logging.info(f"MRFB Loss:\n{self.mrfb_loss}")

        self.discrim_adv_loss = AudioDiscriminatorAdvLoss()
        self.gen_adv_loss = AudioGeneratorAdvLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
    def prepare_models_for_training(self) -> None:
        """Initializes optimizers, schedulers, and SWA for both DAC and discriminator models.

        Uses the `_prepare_model_for_training` helper for both models and sets up
        the `grad_scaler` for mixed precision.
        """
        (
            self.dac_model,
            self.dac_optimizer,
            self.dac_lr_scheduler,
            self.dac_wd_scheduler,
            self.swa_dac_model,
            self.swa_dac_scheduler,
        ) = self._prepare_model_for_training(
            self.dac_model,
            self.dac_optim,
            self.dac_lrsched,
            self.dac_wdsched,
            device=self.device,
            ddp=self.ddp,
            ddp_type=self.ddp_type,
            fsdp_cpu_offload=self.fsdp_cpu_offload,
            fsdp_reshard_after_forward=self.fsdp_reshard_after_forward,
            fsdp_mp_param_dtype=self.fsdp_mp_param_dtype,
            fsdp_mp_reduce_dtype=self.fsdp_mp_reduce_dtype,
            fsdp_mp_output_dtype=self.fsdp_mp_output_dtype,
            do_swa=self.do_swa,
            swa_lr=self.swa_lr,
            swa_anneal_steps=self.swa_anneal_steps,
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
            device=self.device,
            ddp=self.ddp,
            ddp_type=self.ddp_type,
        )
        self.grad_scaler = self.get_grad_scaler(self.use_amp, self.ddp, self.ddp_type)

    def set_train_mode(self) -> None:
        """Applies the selected training modes to the generator and discriminator.

        Also logs parameter summaries and parameter lists if running on rank 0.
        """
        self.dac_model.set_train_mode(self.dac_train_mode)
        self.discrim_model.set_train_mode(self.discrim_train_mode)
        if self.rank == 0:
            logging.info(f"DAC model train mode: {self.dac_train_mode}")
            logging.info(f"Parameter summary for DAC model:")
            self.dac_model.parameter_summary(verbose=True)
            logging.info(f"DAC model parameter list:")
            self.dac_model.print_parameter_list()
            logging.info(f"Discrim model train mode: {self.discrim_train_mode}")
            logging.info(f"Parameter summary for Discrim model:")
            self.discrim_model.parameter_summary(verbose=True)
            logging.info(f"Discrim model parameter list:")
            self.discrim_model.print_parameter_list()

    def on_epoch_begin(self) -> None:
        """Called at the beginning of an epoch.

        Updates all schedulers for both generator and discriminator.
        """
        super().on_epoch_begin()

        for sch in [self.dac_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None:
                sch.on_epoch_begin(self.cur_epoch, save_steps=self.save_steps)

        for sch in [self.dac_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_epoch_begin(self.cur_epoch)

    def on_epoch_end(self, logs: Dict[str, Any]) -> None:
        """Called at the end of an epoch.

        Args:
            logs: Aggregated training logs for the epoch.
        """
        super().on_epoch_end(logs)
        if self.do_swa and self.cur_step >= self.swa_start:
            return

        for sch in [self.dac_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None:
                sch.on_epoch_end(logs)

        for sch in [self.dac_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_epoch_end()

    def on_swa_epoch_begin(self) -> None:
        """Called at the beginning of an SWA epoch.

        Swaps the current VC model with the averaged SWA model.
        """
        super().on_swa_epoch_begin()
        self.dac_model = self.swa_dac_model.module

    def on_swa_epoch_end(self, logs: Dict[str, Any]) -> None:
        """Called at the end of an SWA epoch.

        Args:
            logs: Aggregated training logs for the epoch.
        """
        super().on_swa_epoch_end(logs)

    def on_training_loop_begin(self) -> None:
        """Sets models to training mode before beginning the training loop."""
        self.dac_model.train()
        self.discrim_model.train()

    def on_val_loop_begin(self) -> None:
        """Sets models to evaluation mode before starting validation."""
        self.dac_model.eval()
        self.discrim_model.eval()
        self.cur_val_log_samples = 0

    def preprocess_train_data(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """Prepares and renames training batch data into a standardized format.

        Args:
            batch_data: Raw input batch dictionary.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch.
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

    def preprocess_val_data(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """Prepares validation data using the same layout as training data.

        Args:
            batch_data: Raw input batch dictionary.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and processed batch.
        """
        return self.preprocess_train_data(batch_data)

    def trim_audios(
        self,
        audios: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Matches output and target lengths by trimming context if needed.

        Args:
            audios: Input audio tensor.
            audio_lengths: Optional valid-length tensor for `audios`.

        Returns:
            Tuple[Tensor, Optional[Tensor]]: Trimmed audio and lengths.
        """
        if self.context_trim_fraction > 0.0:
            audios = audios[
                ...,
                self._left_contest_trim : audios.size(-1) - self._right_contest_trim,
            ]
            if audio_lengths is not None:
                audio_lengths = audio_lengths - self._left_contest_trim
                audio_lengths = torch.clamp(audio_lengths, min=0, max=audios.size(-1))

        return audios, audio_lengths

    def training_step(
        self, batch_idx: int, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """Performs the forward and backward passes for both discriminator and generator.

        Handles discriminator training first with real/fake inputs, then updates
        the generator using adversarial and auxiliary losses.

        Args:
            batch_idx: Batch index within the current epoch.
            batch_data: Raw batch dictionary.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and computed metrics.
        """
        batch_size, batch_data = self.preprocess_train_data(batch_data)
        batch_data = self.send_data_to_device(batch_data)
        self.dac_model.update_quantizer_hyperparams(self.cur_step)
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
            # tl1 = target_audios.size(1)
            target_audios, target_lengths = self.dac_model.get_target_matching_output(
                target_audios,
                target_lengths,
            )
            max_recons_length = target_audios.size(-1)
            target_audios, target_lengths = self.trim_audios(
                target_audios,
                target_lengths,
            )

            # tl2 = target_audios.size(-1)

        # il1 = input_audios.size(1)
        # il2 = (
        #     math.ceil(input_audios.size(1) / self.dac_model.encoder.stride)
        #     * self.dac_model.encoder.stride
        # )
        # el1 = self.dac_model.max_out_length(il1)
        # el2 = self.dac_model.max_out_length(il2)
        batch_device_type = input_audios.device.type
        with amp.autocast(
            enabled=self.use_amp, dtype=self.amp_dtype, device_type=batch_device_type
        ):
            dac_output = self.dac_model(
                x=input_audios,
                x_lengths=input_lengths,
            )
            # print(
            #     "lengths",
            #     il1,
            #     il2,
            #     el1,
            #     el2,
            #     tl1,
            #     tl2,
            #     dac_output.x_recons.size(-1),
            #     self.dac_model.delay,
            #     self.dac_model.out_lengths(torch.tensor([32000])).item(),
            #     dac_output.vq.z_q.size(),
            #     self.dac_model.encoder.max_out_length(32000),
            #     self.dac_model.encoder.out_lengths(torch.tensor([32000])).item(),
            #     flush=True,
            # )
            # print("dac_output lengths:", dac_output.x_recons.size(-1), flush=True)
            x_recons, _ = self.trim_audios(
                dac_output.x_recons[..., :max_recons_length],
            )
            # print("x_recons lengths after trimming:", x_recons.size(-1), flush=True)
            y_real, _ = self.discrim_model(
                target_audios,
            )
            y_gen, _ = self.discrim_model(x_recons.detach())

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
            enabled=self.use_amp, dtype=self.amp_dtype, device_type=batch_device_type
        ):
            y_real, fmaps_real = self.discrim_model(target_audios)
            y_gen, fmaps_gen = self.discrim_model(x_recons)

        loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
        loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
        loss_mrfb_log_mag, loss_mrfb_conv = self.mrfb_loss(
            x_recons.squeeze(1), target_audios
        )
        loss_codebook = dac_output.vq.codebook_loss
        loss_commitment = dac_output.vq.commitment_loss
        loss_orthogonality = dac_output.vq.orthogonality_loss
        loss_diversity = dac_output.vq.diversity_loss
        ppl = dac_output.vq.perplexity.mean().detach().item()
        if self.cur_step < self.gen_adv_losses_start_steps:
            loss_gen_adv_weight = 0.0
        elif self.gen_adv_losses_warmup_steps > 0:
            loss_gen_adv_weight = min(
                (self.cur_step - self.gen_adv_losses_start_steps)
                / self.gen_adv_losses_warmup_steps,
                1.0,
            )
        else:
            loss_gen_adv_weight = 1.0
        loss_gen = (
            loss_gen_adv_weight * self.loss_gen_adv_weight * loss_gen_adv
            + loss_gen_adv_weight * self.loss_fm_weight * loss_fm
            + self.loss_mrfb_log_mag_weight * loss_mrfb_log_mag
            + self.loss_mrfb_conv_weight * loss_mrfb_conv
            + self.loss_codebook_weight * loss_codebook
            + self.loss_commitment_weight * loss_commitment
        ) / self.grad_acc_steps
        if self.loss_orthogonality_weight > 0 and loss_orthogonality is not None:
            loss_gen = loss_gen + (
                self.loss_orthogonality_weight
                * loss_orthogonality
                / self.grad_acc_steps
            )
        if self.loss_diversity_weight > 0 and loss_diversity is not None:
            loss_gen = loss_gen + (
                self.loss_diversity_weight * loss_diversity / self.grad_acc_steps
            )

        self.grad_scaler.scale(loss_gen).backward()

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total"] = loss_discrim.item() * self.grad_acc_steps
        batch_metrics["loss_gen/total"] = loss_gen.item() * self.grad_acc_steps
        batch_metrics["loss_gen/mrfb_log_mag"] = loss_mrfb_log_mag.item()
        batch_metrics["loss_gen/mrfb_conv"] = loss_mrfb_conv.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv"] = loss_gen_adv.item()
        batch_metrics["loss_gen/codebook"] = loss_codebook.item()
        batch_metrics["loss_gen/commitment"] = loss_commitment.item()
        if loss_orthogonality is not None:
            batch_metrics["loss_gen/orthogonality"] = loss_orthogonality.item()
        if loss_diversity is not None:
            batch_metrics["loss_gen/diversity"] = loss_diversity.item()

        batch_metrics["loss_gen/ppl_avg"] = ppl
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv/{i}"] = loss

        for i, ppl_i in enumerate(dac_output.vq.perplexity):
            batch_metrics[f"ppl/{i}"] = ppl_i.item()

        return batch_size, batch_metrics

    def validation_step(
        self, batch_idx: int, batch_data: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        """Runs a forward pass through the generator and discriminator during validation.

        Logs spectrograms and audio samples, and computes validation losses.

        Args:
            batch_idx: Batch index within the current epoch.
            batch_data: Raw batch dictionary.

        Returns:
            Tuple[int, Dict[str, Any]]: Batch size and metrics.
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
        # with torch.no_grad():
        target_audios, target_lengths = self.dac_model.get_target_matching_output(
            target_audios,
            target_lengths,
        )
        max_recons_length = target_audios.size(-1)
        target_audios, target_lengths = self.trim_audios(
            target_audios,
            target_lengths,
        )
        # target_audios_sliced, slice_start_idxs = rand_slice_audio_segments(
        #     target_audios,
        #     target_lengths,
        #     self.gen_segment_duration,
        #     self.dac_model.output_sample_frequency,
        # )

        batch_device_type = input_audios.device.type
        with amp.autocast(
            enabled=self.use_amp, dtype=self.amp_dtype, device_type=batch_device_type
        ):
            dac_output = self.dac_model(
                x=input_audios,
                x_lengths=input_lengths,
            )
            x_recons, _ = self.trim_audios(
                dac_output.x_recons[..., :max_recons_length],
            )
            y_real, fmaps_real = self.discrim_model(target_audios)
            y_gen, fmaps_gen = self.discrim_model(x_recons)

        loss_discrim, losses_discrim_adv_gen, losses_discrim_adv_real = (
            self.discrim_adv_loss(y_gen, y_real)
        )
        loss_gen_adv, losses_gen_adv = self.gen_adv_loss(y_gen)
        loss_fm = self.feat_matching_loss(fmaps_gen, fmaps_real)
        loss_mrfb_log_mag, loss_mrfb_conv = self.mrfb_loss(
            x_recons.squeeze(1), target_audios
        )
        loss_codebook = dac_output.vq.codebook_loss
        loss_commitment = dac_output.vq.commitment_loss
        loss_orthogonality = dac_output.vq.orthogonality_loss
        loss_diversity = dac_output.vq.diversity_loss

        ppl = dac_output.vq.perplexity.mean().item()
        if self.cur_step < self.gen_adv_losses_start_steps:
            loss_gen_adv_weight = 0.0
        elif self.gen_adv_losses_warmup_steps > 0:
            loss_gen_adv_weight = min(
                (self.cur_step - self.gen_adv_losses_start_steps)
                / self.gen_adv_losses_warmup_steps,
                1.0,
            )
        else:
            loss_gen_adv_weight = 1.0

        loss_gen = (
            loss_gen_adv_weight * self.loss_gen_adv_weight * loss_gen_adv
            + loss_gen_adv_weight * self.loss_fm_weight * loss_fm
            + self.loss_mrfb_log_mag_weight * loss_mrfb_log_mag
            + self.loss_mrfb_conv_weight * loss_mrfb_conv
            + self.loss_codebook_weight * loss_codebook
            + self.loss_commitment_weight * loss_commitment
        )
        if self.loss_orthogonality_weight > 0 and loss_orthogonality is not None:
            loss_gen = loss_gen + self.loss_orthogonality_weight * loss_orthogonality
        if self.loss_diversity_weight > 0 and loss_diversity is not None:
            loss_gen = loss_gen + self.loss_diversity_weight * loss_diversity

        batch_metrics = ODict()
        batch_metrics["loss_discrim/total"] = loss_discrim.item()
        batch_metrics["loss_gen/total"] = loss_gen.item()
        batch_metrics["loss_gen/mrfb_log_mag"] = loss_mrfb_log_mag.item()
        batch_metrics["loss_gen/mrfb_conv"] = loss_mrfb_conv.item()
        batch_metrics["loss_gen/fm"] = loss_fm.item()
        batch_metrics["loss_gen/adv"] = loss_gen_adv.item()
        batch_metrics["loss_gen/codebook"] = loss_codebook.item()
        batch_metrics["loss_gen/commitment"] = loss_commitment.item()
        if loss_orthogonality is not None:
            batch_metrics["loss_gen/orthogonality"] = loss_orthogonality.item()

        if loss_diversity is not None:
            batch_metrics["loss_gen/diversity"] = loss_diversity.item()

        batch_metrics["loss_gen/ppl_avg"] = ppl
        for i, loss in enumerate(losses_discrim_adv_gen):
            batch_metrics[f"loss_discrim_adv_gen/{i}"] = loss

        for i, loss in enumerate(losses_discrim_adv_real):
            batch_metrics[f"loss_discrim_adv_real/{i}"] = loss

        for i, loss in enumerate(losses_gen_adv):
            batch_metrics[f"loss_gen_adv/{i}"] = loss

        for i, ppl_i in enumerate(dac_output.vq.perplexity):
            batch_metrics[f"ppl/{i}"] = ppl_i.item()

        num_log_samples = min(
            self.num_val_log_samples - self.cur_val_log_samples, batch_size
        )
        # print(
        #     "vallog",
        #     self.num_val_log_samples,
        #     self.cur_val_log_samples,
        #     num_log_samples,
        #     flush=True,
        # )
        for i in range(num_log_samples):
            _id = batch_data["id"][i]
            self.loggers.log_audio(
                f"audios_target/{_id}",
                target_audios[i],
                sample_freq=self.dac_model.output_sample_frequency,
            )
            self.loggers.log_audio(
                f"audios_generated/{_id}",
                dac_output.x_recons[i, 0],
                sample_freq=self.dac_model.output_sample_frequency,
            )
            # self.loggers.log_spectrogram(
            #     f"log_mel_fbanks_target/{_id}", mel_feats_real[i]
            # )
            # self.loggers.log_spectrogram(
            #     f"log_mel_fbanks_generated/{_id}", mel_feats_gen[i]
            # )

        self.cur_val_log_samples += num_log_samples

        return batch_size, batch_metrics

    def update_swa_model(self) -> None:
        """Updates the SWA model parameters and learning rate scheduler if applicable."""
        if (
            self.do_swa
            and self.cur_step >= self.swa_start
            and self.cur_step % self.swa_update_steps == 0
        ):
            self.in_swa = True
            self.swa_dac_model.update_parameters(self.dac_model)
            self.swa_dac_scheduler.step()

    def zero_grad_optimizers(self) -> None:
        """Zeros the gradients for both generator and discriminator optimizers."""
        self.dac_optimizer.zero_grad()
        self.discrim_optimizer.zero_grad()

    def get_lrs(self) -> Dict[str, float]:
        """Returns a dictionary of learning rates for all optimizers."""
        dac_lrs = self._get_lrs(self.dac_optimizer)
        discrim_lrs = self._get_lrs(self.discrim_optimizer)
        lrs = {f"dac_{k}": v for k, v in dac_lrs.items()}
        discrim_lrs = {f"discrim_{k}": v for k, v in discrim_lrs.items()}
        lrs.update(discrim_lrs)
        return lrs

    def get_wds(self) -> Dict[str, float]:
        """Returns a dictionary of weight decay values for all optimizers."""
        dac_wds = self._get_wds(self.dac_optimizer, self.dac_wd_scheduler)
        discrim_wds = self._get_wds(self.discrim_optimizer, self.discrim_wd_scheduler)
        wds = {f"dac_{k}": v for k, v in dac_wds.items()}
        wds.update({f"discrim_{k}": v for k, v in discrim_wds.items()})
        return wds

    def models_have_bn(self) -> bool:
        """Checks if the generator model has any batch normalization layers.

        Returns:
            bool: ``True`` if the generator contains batch normalization layers.
        """
        return self.dac_model.has_batchnorms()

    def update_models(self) -> Dict[str, float]:
        """Steps optimizers and schedulers for both generator and discriminator.

        Also clips gradients and returns logs for gradient norms.

        Returns:
            Dict[str, float]: Gradient norm logs.
        """

        for sch in [self.dac_lr_scheduler, self.discrim_lr_scheduler]:
            if sch is not None and not self.in_swa:
                sch.on_opt_step()

        for sch in [self.dac_wd_scheduler, self.discrim_wd_scheduler]:
            if sch is not None:
                sch.on_opt_step()

        dac_grad_norm = self._update_model_by_optim(
            self.dac_model,
            self.dac_optimizer,
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

        logs = {"grad_norm/dac": dac_grad_norm, "grad_norm/discrim": discrim_grad_norm}
        return logs

    def save_checkpoint(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Saves current training state to disk, including both models and optionally SWA.

        Args:
            logs: Logging metrics to include in the checkpoint.
        """
        if self.rank != 0 and not self.is_fsdp_training():
            return

        checkpoint = self.model_checkpoint(
            self.dac_model,
            self.dac_optimizer,
            self.dac_lr_scheduler,
            self.dac_wd_scheduler,
            self.swa_dac_model,
            self.swa_dac_scheduler,
            logs=logs,
        )

        trainer_checkpoint = checkpoint
        checkpoint = self.model_checkpoint(
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_lr_scheduler,
            self.discrim_wd_scheduler,
            logs=logs,
        )

        if self.rank != 0:
            return

        with self.checkpoint_save_dir(self.cur_epoch, self.cur_step) as checkpoint_dir:
            self.save_model_checkpoint_to_dir(
                checkpoint_dir, "dac_model", trainer_checkpoint
            )
            self.save_model_checkpoint_to_dir(
                checkpoint_dir, "discrim_model", checkpoint
            )
            self.save_trainer_state_to_dir(checkpoint_dir, trainer_checkpoint)

    def save_swa_model(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Saves the final SWA-averaged generator model to disk.

        Args:
            logs: Logging metrics to include in the checkpoint.
        """
        if self.rank != 0 and not self.is_fsdp_training():
            return

        checkpoint = self.swa_model_checkpoint(
            self.dac_model,
            self.swa_dac_model,
        )

        if self.rank != 0:
            return

        self.save_swa_model_to_dir("dac_model", checkpoint)

    def load_checkpoint(self, epoch: int, step: int) -> Optional[Dict[str, Any]]:
        """Loads training state from checkpoint files for both generator and discriminator.

        Args:
            epoch (int): Epoch number of checkpoint.
            step (int): Step number of checkpoint.

        Returns:
            Optional[Dict[str, Any]]: Logs saved with the checkpoint, if any.
        """
        checkpoint_dir = self.checkpoint_dir(epoch, step)
        trainer_state = self.load_trainer_state_from_dir(checkpoint_dir)
        checkpoint = self.load_model_checkpoint_from_dir(checkpoint_dir, "dac_model")
        logs = self._load_vars_from_checkpoint(trainer_state)
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.dac_model,
            self.dac_optimizer,
            self.dac_lr_scheduler,
            self.dac_wd_scheduler,
            self.swa_dac_model,
            self.swa_dac_scheduler,
        )
        checkpoint = self.load_model_checkpoint_from_dir(
            checkpoint_dir, "discrim_model"
        )
        self._load_model_state_dicts_from_checkpoint(
            checkpoint,
            self.discrim_model,
            self.discrim_optimizer,
            self.discrim_lr_scheduler,
            self.discrim_wd_scheduler,
        )
        return logs

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """
        Filters the provided keyword arguments to retain only those valid for the DACTrainer constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Filtered keyword arguments accepted by the constructor.
        """
        args = filter_func_args(DACTrainer.__init__, kwargs)
        return args

    @staticmethod
    def add_optim_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds command-line arguments for generator and discriminator optimizers and schedulers.

        Args:
            parser: Argument parser instance to which arguments are added.
            prefix: Optional namespace prefix to encapsulate arguments.
            skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "dac_optim" not in skip:
            OF.add_class_args(parser, prefix="dac_optim")
        if "dac_lrsched" not in skip:
            LRSF.add_class_args(parser, prefix="dac_lrsched")
        if "dac_wdsched" not in skip:
            WDSF.add_class_args(parser, prefix="dac_wdsched")
        if "discrim_optim" not in skip:
            OF.add_class_args(parser, prefix="discrim_optim")
        if "discrim_lrsched" not in skip:
            LRSF.add_class_args(parser, prefix="discrim_lrsched")
        if "discrim_wdsched" not in skip:
            WDSF.add_class_args(parser, prefix="discrim_wdsched")
        if "discrim_grad_clip" not in skip:
            parser.add_argument(
                "--discrim-grad-clip",
                default=0,
                type=float,
                help="Max norm for clipping discriminator gradients (0 for no clipping).",
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_train_modes_args(
        parser: ArgumentParser, skip: Optional[Set[str]] = None
    ) -> None:
        """
        Adds command-line arguments for generator and discriminator train modes.

        Args:
            parser: Argument parser instance to which arguments are added.
            skip: Argument names to skip.
        """
        if skip is None:
            skip = set()

        if "dac_train_mode" not in skip:
            train_modes = DACTrainMode.choices()
            parser.add_argument(
                "--dac-train-mode",
                default=DACTrainMode.FULL.value,
                choices=train_modes,
                help=(
                    f"Training mode for the generator. "
                    f"Available options: {train_modes}."
                ),
            )
        if "discrim_train_mode" not in skip:
            train_modes = AudioDiscriminatorTrainMode.choices()
            parser.add_argument(
                "--discrim-train-mode",
                default=AudioDiscriminatorTrainMode.FULL.value,
                choices=train_modes,
                help=(
                    f"Training mode for the discriminator. "
                    f"Available options: {train_modes}."
                ),
            )

    @staticmethod
    def add_io_keys_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds command-line arguments to specify batch dictionary keys for input and target audio.

        Args:
            parser: Argument parser to which arguments are added.
            prefix: Optional namespace prefix.
            skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "input_audio_key" not in skip:
            parser.add_argument(
                "--input-audio-key",
                default="audio",
                help="Key used to access source audio in the batch dictionary.",
            )
        if "target_audio_key" not in skip:
            parser.add_argument(
                "--target-audio-key",
                default="audio",
                help="Key used to access target audio in the batch dictionary.",
            )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_loss_weights_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds command-line arguments to configure loss weights for the generator.

        Args:
            parser: Argument parser to which arguments are added.
            prefix: Optional namespace prefix.
            skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        if "loss_mrfb_log_mag_weight" not in skip:
            parser.add_argument(
                "--loss-mrfb-log-mag-weight",
                default=45.0,
                type=float,
                help="Weight for the mel-spectrogram reconstruction loss.",
            )

        if "loss_mrfb_conv_weight" not in skip:
            parser.add_argument(
                "--loss-mrfb-conv-weight",
                default=1.0,
                type=float,
                help="Weight for the mel-spectrogram reconstruction loss.",
            )

        if "loss_fm_weight" not in skip:
            parser.add_argument(
                "--loss-fm-weight",
                default=2.0,
                type=float,
                help="Weight for the discriminator feature-matching loss.",
            )
        if "loss_gen_adv_weight" not in skip:
            parser.add_argument(
                "--loss-gen-adv-weight",
                default=1.0,
                type=float,
                help="Weight for the adversarial generator loss.",
            )

        if "loss_codebook_weight" not in skip:
            parser.add_argument(
                "--loss-codebook-weight",
                default=1.0,
                type=float,
                help="Weight for the codebook loss.",
            )
        if "loss_commitment_weight" not in skip:
            parser.add_argument(
                "--loss-commitment-weight",
                default=0.25,
                type=float,
                help="Weight for the commitment loss.",
            )
        if "loss_orthogonality_weight" not in skip:
            parser.add_argument(
                "--loss-orthogonality-weight",
                default=0.0,
                type=float,
                help="Weight for the orthogonality loss.",
            )
        if "loss_diversity_weight" not in skip:
            parser.add_argument(
                "--loss-diversity-weight",
                default=0.0,
                type=float,
                help="Weight for the diversity loss.",
            )
        if "gen_adv_losses_warmup_steps" not in skip:
            parser.add_argument(
                "--gen-adv-losses-warmup-steps",
                default=0,
                type=int,
                help=(
                    "Number of steps to warm up the adversarial and feature matching losses "
                    "after the start steps."
                ),
            )
        if "gen_adv_losses_start_steps" not in skip:
            parser.add_argument(
                "--gen-adv-losses-start-steps",
                default=0,
                type=int,
                help=(
                    "Number of steps to keep adversarial and feature matching loss weights at 0 "
                    "before the warmup."
                ),
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """
        Adds all DACTrainer-related arguments to the parser, including trainer, optimizer, I/O, and loss configuration.

        Args:
            parser: Argument parser to which arguments are added.
            prefix: Optional prefix to namespace all arguments.
            skip: Set of argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set()

        TorchTrainerBase.add_class_args(parser, skip=skip)
        if "mrfb_loss" not in skip:
            MultiResolutionFilterBankLoss.add_class_args(parser, prefix="mrfb_loss")
        DACTrainer.add_optim_args(parser, skip=skip)
        DACTrainer.add_io_keys_args(parser, skip=skip)
        DACTrainer.add_train_modes_args(parser, skip=skip)
        DACTrainer.add_loss_weights_args(parser, skip=skip)

        if "num_val_log_samples" not in skip:
            parser.add_argument(
                "--num-val-log-samples",
                default=10,
                type=int,
                help="Number of samples to log during validation (audio + spectrogram).",
            )
        if "context_trim_fraction" not in skip:
            parser.add_argument(
                "--context-trim-fraction",
                default=0.0,
                type=float,
                help=(
                    "Fraction of receptive-field context to drop from both ends before "
                    "computing losses."
                ),
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

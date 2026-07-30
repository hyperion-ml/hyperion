"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    import pynvml
except:
    pynvml = None

import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ...utils import PathLike
from .logger import Logger


class TensorBoardLogger(Logger):
    """Logger that sends training progress to tensorboard

    Attributes:
        tb_path: tensorboard output directory
        interval: number of model updates between logs
        gpu_usage: whether to log GPU usage statistics
    """

    def __init__(
        self, tb_path: PathLike, interval: int = 1000, gpu_usage: bool = False
    ) -> None:
        """Initializes TensorBoard writers configuration.

        Args:
            tb_path: output directory for TensorBoard event files.
            interval: logging interval (in steps) for step-level metrics.
            gpu_usage: whether to log GPU memory/utilization/temperature.
        """
        super().__init__()
        self.tb_path = Path(tb_path)
        self.train_writer: Optional[SummaryWriter] = None
        self.val_writer: Optional[SummaryWriter] = None
        if interval <= 0:
            raise ValueError("TensorBoardLogger requires interval > 0")
        self.interval = interval
        self.batches = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        if gpu_usage and pynvml is None:
            logging.warning(
                "[TensorBoardLogger] pynvml is not installed. GPU usage logging will be disabled."
            )
            gpu_usage = False

        self.gpu_usage = gpu_usage
        self._nvml_initialized = False
        self._last_batch_logged_step = -1

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Creates train/validation writers and initializes NVML if requested.

        Args:
            logs: Optional training logs.
            kwargs: Additional callback arguments.
        """
        super().on_train_begin(logs, **kwargs)
        if self.gpu_usage:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except pynvml.NVMLError as err:
                logging.warning("[TensorBoardLogger] NVML init error: %s", err)

        if self.rank != 0:
            return

        purge_step = self.cur_step if self.cur_step > 0 else None
        self.train_writer = SummaryWriter(self.tb_path / "train", purge_step=purge_step)
        self.val_writer = SummaryWriter(self.tb_path / "val", purge_step=purge_step)

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Updates epoch counters and expected batch count.

        Args:
            epoch: Zero-based epoch index.
            logs: Optional logs dictionary.
            kwargs: Additional callback arguments such as ``batches``.
        """
        if self.rank != 0:
            return

        self.cur_epoch = epoch
        if "batches" in kwargs:
            self.batches = kwargs["batches"]
        else:
            self.batches = 0

        self.cur_batch = 0

    def on_batch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs step-level training scalars at the configured interval.

        Args:
            logs: Optional metric dictionary for the current update.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        if (
            self.cur_step % self.interval
        ) == 0 and self.cur_step != self._last_batch_logged_step:
            for k, v in (logs or {}).items():
                self.train_writer.add_scalar(k, v, self.cur_step)
            self._last_batch_logged_step = self.cur_step

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs epoch metrics to train/validation TensorBoard streams.

        Args:
            logs: Optional epoch metric dictionary.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        for k, v in (logs or {}).items():
            if k.startswith("val_"):
                k = k.removeprefix("val_")
                self.val_writer.add_scalar(k, v, self.cur_step)
            else:
                k = k.removeprefix("train_")
                self.train_writer.add_scalar(k, v, self.cur_step)

        self.train_writer.add_scalar("epoch", self.cur_epoch + 1, self.cur_step)

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        self.on_epoch_end(logs, **kwargs)

    def on_train_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Closes writers and shuts down NVML when training completes.

        Args:
            logs: Optional final training logs.
            kwargs: Additional callback arguments.
        """
        if self.gpu_usage and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as err:
                logging.warning("[TensorBoardLogger] NVML shutdown error: %s", err)
            finally:
                self._nvml_initialized = False

        if self.rank != 0:
            return
        self.train_writer.close()
        self.val_writer.close()
        self.train_writer = None
        self.val_writer = None

    def log_model_weights_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of selected model weights to TensorBoard.

        Args:
            model (nn.Module): The model to log.
            pattern (str): Regular expression to filter parameter names.
                           Only matching parameters will be logged.
                           Default ".*" logs all parameters.
        """
        if self.rank != 0 or self.train_writer is None:
            return

        regex = re.compile(pattern)

        for name, param in model.named_parameters():
            if param.requires_grad and regex.match(name):
                self.train_writer.add_histogram(
                    name, param.data.cpu().numpy(), self.cur_step
                )

    def log_model_gradients_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of selected model gradients to TensorBoard.

        Args:
            model (nn.Module): The model to log.
            pattern (str): Regular expression to filter parameter names.
                           Only matching parameters with gradients will be logged.
                           Default ".*" logs all gradients.
        """
        if self.rank != 0 or self.train_writer is None:
            return

        regex = re.compile(pattern)

        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad and regex.match(name):
                self.train_writer.add_histogram(
                    f"{name}_grad", param.grad.data.cpu().numpy(), self.cur_step
                )

    def log_audio(
        self,
        tag: str,
        audio: torch.Tensor,
        sample_freq: int,
        phase: str = "val",
    ) -> None:
        """Logs a single audio sample to TensorBoard.

        Args:
            tag (str): Name of the audio tag (e.g., "generated", "reconstructed").
            audio (torch.Tensor): Audio tensor of shape (samples,) or (1, samples). Should be 1D or 2D with single channel.
            sample_freq (int): Sample rate in Hz (e.g., 16000).
            phase (str): Logging stream, "train" or "val".
        """
        if self.rank != 0 or self.train_writer is None:
            return

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Make it (1, samples)

        if audio.dtype != torch.float32:
            audio = audio.float()

        if phase == "val":
            writer = self.val_writer
        else:
            writer = self.train_writer

        writer.add_audio(tag, audio, global_step=self.cur_step, sample_rate=sample_freq)

    def log_spectrogram(
        self,
        tag: str,
        spec: torch.Tensor,
        cmap: str = "plasma",
        apply_log: bool = False,
        phase: str = "val",
    ) -> None:
        """Logs a log-magnitude spectrogram image to TensorBoard.

        Args:
            tag (str): Tag name for the spectrogram (e.g., "mel_pred", "mel_target").
            spec (torch.Tensor): Spectrogram tensor of shape (time_frames, freq_bins) or (1, time_frames, freq_bins).
            cmap (str): Colormap used in matplotlib (e.g., "viridis", "magma", "plasma").
            apply_log (bool): Whether to apply log scaling before plotting.
            phase (str): Logging stream, "train" or "val".
        """
        if self.rank != 0 or self.train_writer is None:
            return

        step = self.cur_step

        if spec.dim() == 3 and spec.size(0) == 1:
            spec = spec.squeeze(0)

        if spec.dim() != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")

        # Convert to log scale if not already
        if apply_log:
            spec = torch.clamp(spec, min=1e-5).log()

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            spec.transpose(-1, -2).cpu().numpy(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )
        ax.set_title(tag)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        # Save plot to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        # Convert buffer to image
        import PIL.Image

        image = PIL.Image.open(buf)
        image = np.array(image)

        if phase == "val":
            writer = self.val_writer
        else:
            writer = self.train_writer
        writer.add_image(tag, image, global_step=step, dataformats="HWC")

    def log_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        dataformats: str = "HWC",
        phase: str = "val",
    ) -> None:
        """Logs a single image to TensorBoard.

        Args:
            tag (str): Name/tag for the image (e.g., "output", "attention_map").
            image (np.ndarray | torch.Tensor | PIL.Image.Image): Image to log.
                Must be in HWC or CHW format depending on `dataformats`.
            dataformats (str): TensorBoard image data format, usually "HWC" or "CHW".
            phase (str): Logging stream, "train" or "val".
        """
        if self.rank != 0 or self.train_writer is None:
            return

        step = self.cur_step

        # Convert PIL to np.array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert torch to np.array
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        # Convert grayscale 2D to 3D (HWC)
        if image.ndim == 2 and dataformats == "HWC":
            image = np.stack([image] * 3, axis=-1)

        if phase == "val":
            writer = self.val_writer
        else:
            writer = self.train_writer

        writer.add_image(tag, image, global_step=step, dataformats=dataformats)

    def log_attention(
        self,
        tag: str,
        attn: torch.Tensor,
        head_labels: Optional[List[str]] = None,
        max_heads: int = 8,
        phase: str = "val",
    ) -> None:
        """Logs attention maps from Transformer heads to TensorBoard.

        Args:
            tag (str): Base tag for logging (e.g., "encoder_attn").
            attn (torch.Tensor): Attention weights of shape
                - (num_heads, tgt_len, src_len) or
                - (batch_size, num_heads, tgt_len, src_len)

            head_labels (list, optional): Optional list of head names (for better display).
            max_heads (int): Max number of heads to visualize (default 8).
            phase (str): Logging stream, "train" or "val".
        """
        if self.rank != 0 or self.train_writer is None:
            return

        step = self.cur_step

        # Normalize and squeeze batch if necessary
        if attn.dim() == 4:
            attn = attn[0]  # Take the first sample

        if attn.dim() != 3:
            raise ValueError(
                f"Expected attention shape (heads, tgt_len, src_len), got {attn.shape}"
            )

        num_heads = min(attn.size(0), max_heads)

        if phase == "val":
            writer = self.val_writer
        else:
            writer = self.train_writer

        for h in range(num_heads):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(
                attn[h].cpu().numpy(), aspect="auto", origin="lower", cmap="viridis"
            )
            label = (
                head_labels[h] if head_labels and h < len(head_labels) else f"head{h}"
            )
            ax.set_title(f"{tag} - {label}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")
            plt.tight_layout()

            # Save to image
            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            image = np.array(Image.open(buf).convert("RGB"))
            writer.add_image(
                f"{tag}/{label}", image, global_step=step, dataformats="HWC"
            )

    def log_gpu_usage(self, device_index: Optional[int] = None) -> None:
        """Logs GPU usage (memory and utilization) as TensorBoard scalars.

        Args:
            device_index (int): Local GPU index to monitor; if None each rank logs
                only its current CUDA device.
        """
        if not self.gpu_usage:
            return

        if (self.cur_step % self.interval) != 0:
            return

        local_payload: Dict[str, Any] = {"rank": self.rank, "gpus": [], "error": None}

        try:
            num_devices = pynvml.nvmlDeviceGetCount()
            if num_devices <= 0:
                local_payload["error"] = "no_visible_gpus"
            else:
                local_device = device_index
                if local_device is None:
                    local_device = (
                        torch.cuda.current_device() if torch.cuda.is_available() else 0
                    )

                if local_device < 0 or local_device >= num_devices:
                    logging.warning(
                        "[TensorBoardLogger] Invalid local GPU index %d (available=%d)",
                        local_device,
                        num_devices,
                    )
                    local_payload["error"] = "invalid_device_index"
                else:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(local_device)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    local_payload["gpus"].append(
                        {
                            "device_index": local_device,
                            "memory_used_MB": mem_info.used / 1024**2,
                            "memory_total_MB": mem_info.total / 1024**2,
                            "utilization_percent": util.gpu,
                            "temperature_C": temp,
                        }
                    )

        except pynvml.NVMLError as err:
            logging.warning("[TensorBoardLogger] NVML error: %s", err)
            local_payload["error"] = str(err)

        payloads: List[Dict[str, Any]] = [local_payload]
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            payloads = [None for _ in range(self.world_size)]
            dist.all_gather_object(payloads, local_payload)

        if self.rank != 0 or self.train_writer is None:
            return

        step = self.cur_step
        for p in payloads:
            if not p:
                continue
            rank = p["rank"]
            for gpu in p.get("gpus", []):
                prefix = f"gpu/rank_{rank}/device_{gpu['device_index']}"
                self.train_writer.add_scalar(
                    f"{prefix}/memory_used_MB", gpu["memory_used_MB"], step
                )
                self.train_writer.add_scalar(
                    f"{prefix}/memory_total_MB", gpu["memory_total_MB"], step
                )
                self.train_writer.add_scalar(
                    f"{prefix}/utilization_percent", gpu["utilization_percent"], step
                )
                self.train_writer.add_scalar(
                    f"{prefix}/temperature_C", gpu["temperature_C"], step
                )

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs update-level scalars and refreshes current global step.

        Args:
            step: Global optimizer step.
            logs: Optional update-level metrics.
            kwargs: Additional callback arguments.
        """
        super().on_model_update(step, logs, **kwargs)
        if self.gpu_usage:
            self.log_gpu_usage()
        if self.rank != 0 or not logs:
            return

        for k, v in logs.items():
            self.train_writer.add_scalar(k, v, self.cur_step)

"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import os
import re

try:
    import wandb
except:
    pass

import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pynvml
import torch
import wandb
from PIL import Image
from torch import nn

from .logger import Logger


class WAndBLogger(Logger):
    """Logger that sends training progress to Weights & Biases (wandb)."""

    def __init__(
        self,
        project: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        mode: str = "online",
        interval: int = 1000,
        gpu_usage: bool = False,
    ):
        """
        Args:
            project (str): WandB project name.
            group (str): WandB group name (for hyperparameter sweeps or batch runs).
            name (str): Run name.
            path (str | Path): Output directory for wandb logs.
            mode (str): WandB mode: "online", "offline", or "disabled".
            interval (int): Number of steps between logging batches.
        """
        super().__init__()
        self.project = project
        self.path = Path(path) if path is not None else None
        self.name = name
        self.group = group
        self.mode = mode
        self.interval = interval
        self.batches = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        self.gpu_usage = gpu_usage

    def on_train_begin(self, logs: Dict[str, Any] = None, **kwargs):
        """Initializes the wandb run."""
        super().on_train_begin(logs, **kwargs)
        if self.rank != 0:
            return

        if self.gpu_usage:
            pynvml.nvmlInit()

        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

        wandb.init(
            project=self.project,
            group=self.group,
            name=self.name,
            dir=str(self.path) if self.path else None,
            mode=self.mode,
            reinit=True,
        )

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] = None, **kwargs):
        """Updates the current epoch and batch count."""
        if self.rank != 0:
            return
        self.cur_epoch = epoch
        self.batches = kwargs.get("batches", 0)
        self.cur_batch = 0

    def on_batch_end(self, logs: Dict[str, Any] = None, **kwargs):
        """Logs batch metrics at specified intervals."""
        if self.rank != 0:
            return

        if self.cur_step % self.interval == 0:
            logs = {f"train/{k}": v for k, v in (logs or {}).items()}
            logs["batch"] = self.cur_step
            wandb.log(logs, step=self.cur_step)

    def on_epoch_end(self, logs: Dict[str, Any] = None, **kwargs):
        """Logs epoch-level metrics."""
        if self.rank != 0:
            return

        logs = logs or {}
        logs = {re.sub(r"^(train|val)_(.*)$", r"\1/\2", k): v for k, v in logs.items()}
        logs["epoch"] = self.cur_epoch + 1
        wandb.log(logs, step=self.cur_step)

    def on_val_end(self, logs=None, **kwargs):
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        self.on_epoch_end(logs, **kwargs)

    def on_train_end(self, logs: Dict[str, Any] = None, **kwargs):
        """Finalizes the wandb run."""
        if self.rank != 0:
            return

        if self.gpu_usage:
            pynvml.nvmlShutdown()
        wandb.finish()

    def log_model_weights_histograms(self, model: nn.Module, pattern: str = ".*"):
        """Logs histograms of model weights matching a regex pattern."""
        if self.rank != 0:
            return
        regex = re.compile(pattern)
        for name, param in model.named_parameters():
            if param.requires_grad and regex.match(name):
                wandb.log(
                    {f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())},
                    step=self.cur_step,
                )

    def log_model_gradients_histograms(self, model: nn.Module, pattern: str = ".*"):
        """Logs histograms of model gradients matching a regex pattern."""
        if self.rank != 0:
            return
        regex = re.compile(pattern)
        for name, param in model.named_parameters():
            if param.grad is not None and param.requires_grad and regex.match(name):
                wandb.log(
                    {
                        f"gradients/{name}": wandb.Histogram(
                            param.grad.data.cpu().numpy()
                        )
                    },
                    step=self.cur_step,
                )

    def log_audio(
        self, tag: str, audio: torch.Tensor, sample_freq: int, phase: str = "val"
    ):
        """Logs an audio sample.

        Args:
            tag (str): Identifier for the audio.
            audio (Tensor): Audio tensor of shape (samples,) or (1, samples).
            sample_freq (int): Sampling rate in Hz.
            phase (str): One of "train" or "val".
        """
        if self.rank != 0:
            return
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.dtype != torch.float32:
            audio = audio.float()
        wandb.log(
            {
                f"{phase}/audio/{tag}": wandb.Audio(
                    audio.cpu().numpy(), sample_rate=sample_freq
                )
            },
            step=self.cur_step,
        )

    def log_spectrogram(
        self,
        tag: str,
        spec: torch.Tensor,
        cmap: str = "plasma",
        apply_log: bool = False,
        phase: str = "val",
    ):
        """Logs a spectrogram image.

        Args:
            tag (str): Identifier.
            spec (Tensor): 2D spectrogram (or 3D with batch dim 1).
            cmap (str): Colormap for rendering.
            apply_log (bool): Whether to apply log scaling.
            phase (str): "train" or "val".
        """
        if self.rank != 0:
            return
        if spec.dim() == 3 and spec.size(0) == 1:
            spec = spec.squeeze(0)
        if spec.dim() != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")
        if apply_log:
            spec = torch.clamp(spec, min=1e-5).log()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(
            spec.transpose(-1, -2).cpu().numpy(),
            aspect="auto",
            origin="lower",
            cmap=cmap,
        )
        ax.set_title(tag)
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        image = Image.open(buf)
        wandb.log(
            {f"{phase}/spectrogram/{tag}": wandb.Image(image)}, step=self.cur_step
        )

    def log_image(
        self,
        tag: str,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        step: Optional[int] = None,
        dataformats: str = "HWC",
        phase: str = "val",
    ):
        """Logs a single image.

        Args:
            tag (str): Image tag.
            image (ndarray | Tensor | Image): Input image.
            step (int): Optional step override.
            dataformats (str): "HWC" or "CHW".
            phase (str): "train" or "val".
        """
        if self.rank != 0:
            return
        step = self.cur_step if step is None else step
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 2 and dataformats == "HWC":
            image = np.stack([image] * 3, axis=-1)
        wandb.log({f"{phase}/image/{tag}": wandb.Image(image)}, step=step)

    def log_attention(
        self,
        tag: str,
        attn: torch.Tensor,
        head_labels: Optional[list] = None,
        max_heads: int = 8,
        phase: str = "val",
    ):
        """Logs Transformer attention maps.

        Args:
            tag (str): Base tag name.
            attn (Tensor): Shape (num_heads, tgt_len, src_len) or (B, num_heads, tgt_len, src_len).
            head_labels (list): Optional names for heads.
            max_heads (int): Maximum number of heads to log.
            phase (str): "train" or "val".
        """
        if self.rank != 0:
            return
        if attn.dim() == 4:
            attn = attn[0]
        if attn.dim() != 3:
            raise ValueError(
                f"Expected attention shape (heads, tgt_len, src_len), got {attn.shape}"
            )

        num_heads = min(attn.size(0), max_heads)
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

            buf = BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            wandb.log(
                {f"{phase}/attention/{tag}_{label}": wandb.Image(image)},
                step=self.cur_step,
            )

    def log_gpu_usage(self, device_index: Optional[int] = None):
        """Logs GPU usage stats: memory, utilization, temperature.

        Args:
            device_index (int): Index to monitor. If None, logs all GPUs.
        """
        if self.rank != 0:
            return
        if self.cur_step % self.interval != 0:
            return

        try:
            indexes = (
                [device_index] if device_index is not None else range(self.world_size)
            )
            for i in indexes:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                wandb.log(
                    {
                        f"gpu/{i}/memory_used_MB": mem.used / 1024**2,
                        f"gpu/{i}/memory_total_MB": mem.total / 1024**2,
                        f"gpu/{i}/utilization_percent": util.gpu,
                        f"gpu/{i}/temperature_C": temp,
                    },
                    step=self.cur_step,
                )
        except pynvml.NVMLError as err:
            print(f"[WAndBLogger] NVML error: {err}")

    def on_model_update(self, step, logs=None, **kwargs):
        super().on_model_update(step, logs, **kwargs)
        if self.rank != 0 or not logs:
            return

        wandb.log(logs, step=self.cur_step)

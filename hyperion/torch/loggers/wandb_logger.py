"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import re

try:
    import wandb
except Exception:
    wandb = None

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
    ) -> None:
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
        if wandb is None:
            raise ImportError(
                "WAndBLogger requires the optional dependency 'wandb'. "
                "Install wandb or remove WAndBLogger from the logger list."
            )
        self.project = project
        self.path = Path(path) if path is not None else None
        self.name = name
        self.group = group
        self.mode = mode
        if interval <= 0:
            raise ValueError("WAndBLogger requires interval > 0")
        self.interval = interval
        self.batches = 0
        self.cur_epoch = 0
        self.cur_batch = 0
        if gpu_usage and pynvml is None:
            logging.warning(
                "[WAndBLogger] pynvml is not installed. GPU usage logging will be disabled."
            )
            gpu_usage = False

        self.gpu_usage = gpu_usage
        self._nvml_initialized = False
        self._last_batch_logged_step = -1

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Initializes the wandb run.

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
                logging.warning("[WAndBLogger] NVML init error: %s", err)

        if self.rank != 0:
            return

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

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Updates the current epoch and batch count.

        Args:
            epoch: Zero-based epoch index.
            logs: Optional logs dictionary.
            kwargs: Additional callback arguments such as ``batches``.
        """
        if self.rank != 0:
            return
        self.cur_epoch = epoch
        self.batches = kwargs.get("batches", 0)
        self.cur_batch = 0

    def on_batch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs batch metrics at specified intervals.

        Args:
            logs: Optional metric dictionary for the current update.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        if (
            self.cur_step % self.interval == 0
            and self.cur_step != self._last_batch_logged_step
        ):
            logs = {f"train/{k}": v for k, v in (logs or {}).items()}
            logs["batch"] = self.cur_step
            wandb.log(logs, step=self.cur_step)
            self._last_batch_logged_step = self.cur_step

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs epoch-level metrics.

        Args:
            logs: Optional epoch metric dictionary.
            kwargs: Additional callback arguments.
        """
        if self.rank != 0:
            return

        logs = logs or {}
        logs = {re.sub(r"^(train|val)_(.*)$", r"\1/\2", k): v for k, v in logs.items()}
        logs["epoch"] = self.cur_epoch + 1
        wandb.log(logs, step=self.cur_step)

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        self.on_epoch_end(logs, **kwargs)

    def on_train_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Finalizes the wandb run.

        Args:
            logs: Optional final training logs.
            kwargs: Additional callback arguments.
        """
        if self.gpu_usage and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError as err:
                logging.warning("[WAndBLogger] NVML shutdown error: %s", err)
            finally:
                self._nvml_initialized = False

        if self.rank != 0:
            return
        wandb.finish()

    def log_model_weights_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of model weights matching a regex pattern.

        Args:
            model: Model whose parameters will be logged.
            pattern: Regex used to filter parameter names.
        """
        if self.rank != 0:
            return
        regex = re.compile(pattern)
        for name, param in model.named_parameters():
            if param.requires_grad and regex.match(name):
                wandb.log(
                    {f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())},
                    step=self.cur_step,
                )

    def log_model_gradients_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of model gradients matching a regex pattern.

        Args:
            model: Model whose gradients will be logged.
            pattern: Regex used to filter parameter names.
        """
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
    ) -> None:
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
    ) -> None:
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
        dataformats: str = "HWC",
        phase: str = "val",
    ) -> None:
        """Logs a single image.

        Args:
            tag (str): Image tag.
            image (ndarray | Tensor | Image): Input image.
            dataformats (str): "HWC" or "CHW".
            phase (str): "train" or "val".
        """
        if self.rank != 0:
            return
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 2 and dataformats == "HWC":
            image = np.stack([image] * 3, axis=-1)
        wandb.log({f"{phase}/image/{tag}": wandb.Image(image)}, step=self.cur_step)

    def log_attention(
        self,
        tag: str,
        attn: torch.Tensor,
        head_labels: Optional[List[str]] = None,
        max_heads: int = 8,
        phase: str = "val",
    ) -> None:
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

    def log_gpu_usage(self, device_index: Optional[int] = None) -> None:
        """Logs GPU usage stats: memory, utilization, temperature.

        Args:
            device_index (int): Local GPU index to monitor; if None each rank logs
                only its current CUDA device.
        """
        if not self.gpu_usage:
            return
        if self.cur_step % self.interval != 0:
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
                        "[WAndBLogger] Invalid local GPU index %d (available=%d)",
                        local_device,
                        num_devices,
                    )
                    local_payload["error"] = "invalid_device_index"
                else:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(local_device)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    local_payload["gpus"].append(
                        {
                            "device_index": local_device,
                            "memory_used_MB": mem.used / 1024**2,
                            "memory_total_MB": mem.total / 1024**2,
                            "utilization_percent": util.gpu,
                            "temperature_C": temp,
                        }
                    )
        except pynvml.NVMLError as err:
            logging.warning("[WAndBLogger] NVML error: %s", err)
            local_payload["error"] = str(err)

        payloads: List[Dict[str, Any]] = [local_payload]
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            payloads = [None for _ in range(self.world_size)]
            dist.all_gather_object(payloads, local_payload)

        if self.rank != 0:
            return

        logs = {}
        for p in payloads:
            if not p:
                continue
            rank = p["rank"]
            for gpu in p.get("gpus", []):
                prefix = f"gpu/rank_{rank}/device_{gpu['device_index']}"
                logs[f"{prefix}/memory_used_MB"] = gpu["memory_used_MB"]
                logs[f"{prefix}/memory_total_MB"] = gpu["memory_total_MB"]
                logs[f"{prefix}/utilization_percent"] = gpu["utilization_percent"]
                logs[f"{prefix}/temperature_C"] = gpu["temperature_C"]
        if logs:
            wandb.log(logs, step=self.cur_step)

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Logs update-level metrics and refreshes current global step.

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

        wandb.log(logs, step=self.cur_step)

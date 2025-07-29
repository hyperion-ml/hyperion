"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

from .logger import Logger
from .tensorboard_logger import TensorBoardLogger as TBL


class LoggerList:
    """Container for a list of logger callbacks

    Attributes:
       loggers: list of Logger objects
    """

    def __init__(self, loggers: Optional[List[Logger]] = None):
        self.loggers = loggers or []

    def append(self, logger):
        self.loggers.append(logger)

    @property
    def tensorboard_logger(self):
        for l in self.loggers:
            if isinstance(l, TBL):
                return l

    @property
    def tensorboard_writer(self):
        for l in self.loggers:
            if isinstance(l, TBL):
                return l.writer

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """At the start of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_begin(epoch, logs, **kwargs)

    def on_epoch_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """At the end of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_end(logs, **kwargs)

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_val_end(logs, **kwargs)

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """At the start of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_begin(batch, logs, **kwargs)

    def on_batch_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """At the end of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_end(logs, **kwargs)

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """At the end of a model update

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_model_update(step, logs, **kwargs)

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """At the start of training

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_begin(logs, **kwargs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs):
        """At the end of training

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_end(logs, **kwargs)

    def __iter__(self):
        return iter(self.loggers)

    def log_model_weights_histograms(self, model: nn.Module, pattern: str = ".*"):
        """Logs histograms of model weights matching a regex pattern."""
        for logger in self.loggers:
            if hasattr(logger, "log_model_weights_histograms"):
                logger.log_model_weights_histograms(model, pattern)

    def log_model_gradients_histograms(self, model: nn.Module, pattern: str = ".*"):
        """Logs histograms of model gradients matching a regex pattern."""
        for logger in self.loggers:
            if hasattr(logger, "log_model_gradients_histograms"):
                logger.log_model_gradients_histograms(model, pattern)

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
        for logger in self.loggers:
            if hasattr(logger, "log_audio"):
                logger.log_audio(tag, audio, sample_freq, phase)

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
        for logger in self.loggers:
            if hasattr(logger, "log_spectrogram"):
                logger.log_spectrogram(tag, spec, cmap, apply_log, phase)

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
        for logger in self.loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(tag, image, step, dataformats, phase)

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
        for logger in self.loggers:
            if hasattr(logger, "log_attention"):
                logger.log_attention(tag, attn, head_labels, max_heads, phase)

    def log_gpu_usage(self, device_index: Optional[int] = None):
        """Logs GPU usage stats: memory, utilization, temperature.

        Args:
            device_index (int): Index to monitor. If None, logs all GPUs.
        """
        for logger in self.loggers:
            if hasattr(logger, "log_gpu_usage"):
                logger.log_gpu_usage(device_index)

"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .logger import Logger
from .tensorboard_logger import TensorBoardLogger as TBL


class LoggerList:
    """Container for a list of logger callbacks

    Attributes:
       loggers: list of Logger objects
    """

    def __init__(self, loggers: Optional[List[Logger]] = None) -> None:
        """Builds a container of logger callbacks.

        Args:
           loggers: initial list of logger instances.
        """
        self.loggers = loggers or []

    def append(self, logger: Logger) -> None:
        """Appends a logger callback to the list.

        Args:
            logger: Logger instance to append.
        """
        self.loggers.append(logger)

    @property
    def tensorboard_logger(self) -> Optional[TBL]:
        """Returns the first TensorBoard logger if present."""
        for l in self.loggers:
            if isinstance(l, TBL):
                return l

    @property
    def tensorboard_train_writer(self) -> Optional[Any]:
        """Returns the train writer associated with the first TensorBoard logger."""
        for l in self.loggers:
            if isinstance(l, TBL):
                return l.train_writer

    @property
    def tensorboard_val_writer(self) -> Optional[Any]:
        """Returns the validation writer associated with the first TensorBoard logger."""
        for l in self.loggers:
            if isinstance(l, TBL):
                return l.val_writer

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of an epoch

        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_begin(epoch, logs, **kwargs)

    def on_epoch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of an epoch

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_end(logs, **kwargs)

    def on_val_end(self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """At the end of validation

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_val_end(logs, **kwargs)

    def on_batch_begin(
        self, batch: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of a batch

        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_begin(batch, logs, **kwargs)

    def on_batch_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of a batch

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_end(logs, **kwargs)

    def on_model_update(
        self, step: int, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of a model update

        Args:
           step: index of the step
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_model_update(step, logs, **kwargs)

    def on_train_begin(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the start of training

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_begin(logs, **kwargs)

    def on_train_end(
        self, logs: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """At the end of training

        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_end(logs, **kwargs)

    def __iter__(self) -> Iterator[Logger]:
        """Returns an iterator over contained loggers."""
        return iter(self.loggers)

    def log_model_weights_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of model weights matching a regex pattern.

        Args:
            model: Model whose parameters will be logged.
            pattern: Regex used to filter parameter names.
        """
        for logger in self.loggers:
            if hasattr(logger, "log_model_weights_histograms"):
                logger.log_model_weights_histograms(model, pattern)

    def log_model_gradients_histograms(
        self, model: nn.Module, pattern: str = ".*"
    ) -> None:
        """Logs histograms of model gradients matching a regex pattern.

        Args:
            model: Model whose gradients will be logged.
            pattern: Regex used to filter parameter names.
        """
        for logger in self.loggers:
            if hasattr(logger, "log_model_gradients_histograms"):
                logger.log_model_gradients_histograms(model, pattern)

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
    ) -> None:
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
        for logger in self.loggers:
            if hasattr(logger, "log_image"):
                logger.log_image(tag, image, dataformats, phase)

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
        for logger in self.loggers:
            if hasattr(logger, "log_attention"):
                logger.log_attention(tag, attn, head_labels, max_heads, phase)

    def log_gpu_usage(self, device_index: Optional[int] = None) -> None:
        """Logs GPU usage stats: memory, utilization, temperature.

        Args:
            device_index (int): Local GPU index to monitor per rank. If None,
                each logger chooses the current local GPU for that rank.
        """
        for logger in self.loggers:
            if hasattr(logger, "log_gpu_usage"):
                logger.log_gpu_usage(device_index)

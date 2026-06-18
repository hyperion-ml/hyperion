"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layers import GatherDistributedFunction
from ..utils import ddp_get_rank


class ContrastiveLoss(nn.Module):
    """Contrastive loss with distributed gathering and optional negatives.

    Attributes:
        temp: Target temperature used after warmup.
        cur_temp: Current temperature value.
        temp_warmup_steps: Number of steps for temperature warmup.
        initial_temp: Starting temperature value.
        margin: Target margin used after warmup.
        margin_warmup_steps: Number of steps for margin warmup.
        margin_warmup_start: Step at which margin warmup starts.
        cur_margin: Current margin value.
        log_interval: Logging interval for schedule updates.
    """

    def __init__(
        self,
        temp: float = 0.07,
        temp_warmup_steps: int = 1000,
        initial_temp: float = 0.07,
        margin: float = 0.3,
        margin_warmup_steps: int = 0,
        margin_warmup_start: int = 0,
        log_interval: int = 1000,
    ) -> None:
        """Initializes the loss.

        Args:
            temp: Final temperature used after warmup.
            temp_warmup_steps: Number of steps over which to warm up the temperature.
            initial_temp: Temperature used at step zero.
            margin: Final margin used after warmup.
            margin_warmup_steps: Number of steps over which to warm up the margin.
            margin_warmup_start: Step at which margin warmup starts.
            log_interval: Interval for logging schedule updates.
        """
        super().__init__()
        self.temp = temp
        self.cur_temp = initial_temp
        self.temp_warmup_steps = temp_warmup_steps
        self.initial_temp = initial_temp
        self.margin = margin
        self.margin_warmup_steps = margin_warmup_steps
        self.margin_warmup_start = margin_warmup_start
        self.cur_margin = 0.0
        self.log_interval = log_interval

    def update_temp(self, step: int) -> None:
        """Updates the temperature schedule.

        Args:
            step: Current training step.
        """
        if step < self.temp_warmup_steps:
            self.cur_temp = (
                self.initial_temp
                + (self.temp - self.initial_temp) * step / self.temp_warmup_steps
            )
            if step % self.log_interval == 0:
                logging.info("updating contrastive losss temp=%.3f", self.cur_temp)
        else:
            self.cur_temp = self.temp

    def update_margin(self, step: int) -> None:
        """Updates the margin schedule.

        Args:
            step: Current training step.
        """
        if step < self.margin_warmup_start:
            self.cur_margin = 0.0
        elif step < self.margin_warmup_steps + self.margin_warmup_start:
            self.cur_margin = (
                self.margin
                * (step - self.margin_warmup_start)
                / self.margin_warmup_steps
            )
            if step % self.log_interval == 0:
                logging.info(
                    "updating constrastive loss margin=%.2f",
                    self.cur_margin,
                )
        else:
            if self.cur_margin != self.margin:
                self.cur_margin = self.margin
                if step % self.log_interval == 0:
                    logging.info(
                        "updating constrastive loss margin=%.2f",
                        self.cur_margin,
                )

    def update(self, step: int) -> None:
        """Updates both scheduled loss parameters.

        Args:
            step: Current training step.
        """
        self.update_temp(step)
        self.update_margin(step)

    def forward(
        self,
        z_pred: torch.Tensor,
        z_true: torch.Tensor,
        z_negatives: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes the contrastive loss.

        Args:
            z_pred: Predicted embeddings with shape ``(B, D)``.
            z_true: Ground-truth embeddings with shape ``(B, D)``.
            z_negatives: Optional additional negatives with shape ``(K, D)``.
            return_logits: If ``True``, also return the logits tensor.

        Returns:
            Scalar loss tensor, or ``(loss, logits)`` when ``return_logits`` is
            ``True``.
        """
        B, D = z_pred.shape

        # Normalize local features
        z_pred = nn.functional.normalize(z_pred, dim=1)
        z_true = nn.functional.normalize(z_true, dim=1)

        # Gather across distributed GPUs
        z_true_all = GatherDistributedFunction.apply(z_true)  # (G*B, D)

        # Optionally append memory bank z_negatives
        if z_negatives is not None:
            z_true_all = torch.cat(
                [z_true_all, nn.functional.normalize(z_negatives, dim=1)], dim=0
            )

        # Similarity matrices
        logits = torch.matmul(z_pred, z_true_all.T)

        # Labels are position indices: correct match at local position + rank offset
        rank = ddp_get_rank()
        labels = torch.arange(B, device=z_pred.device) + rank * B

        if self.cur_margin > 0.0 and self.training:
            # Apply margin to logits
            logits = logits.clone()
            logits_m = logits - self.cur_margin
            idx_ = torch.arange(
                0, logits.shape[0], dtype=torch.long, device=z_pred.device
            )
            logits[idx_, labels] = logits_m[idx_, labels]

        logits = logits / self.cur_temp  # Scale by temperature

        # Cross-entropy losses
        loss = nn.functional.cross_entropy(logits, labels)
        if return_logits:
            return loss, logits

        return loss

    @staticmethod
    def filter_args(**kwargs: object) -> dict:
        """Filters keyword arguments accepted by ``__init__``.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dictionary containing the accepted keyword arguments.
        """
        return filter_func_args(ContrastiveLoss.__init__, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds CLI arguments for this loss.

        Args:
            parser: Argument parser to extend.
            prefix: Optional nested prefix for grouped arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--temp",
            default=0.1,
            type=float,
            help="Target temperature scaling value for similarity logits",
        )
        parser.add_argument(
            "--initial-temp",
            default=0.04,
            type=float,
            help="Initial temperature used at the start of training before warmup completes",
        )
        parser.add_argument(
            "--temp-warmup-steps",
            default=30,
            type=int,
            help="Number of steps over which to linearly warm up the temperature from `initial_temp` to `temp`",
        )
        parser.add_argument(
            "--margin",
            default=0.1,
            type=float,
            help="Margin value for contrastive loss",
        )
        parser.add_argument(
            "--margin-warmup-steps",
            default=30,
            type=int,
            help="Number of steps over which to linearly warm up the margin from 0 to `margin`",
        )
        parser.add_argument(
            "--margin-warmup-start",
            default=0,
            type=int,
            help="Step at which to start warming up the margin",
        )
        parser.add_argument(
            "--log-interval",
            default=1000,
            type=int,
            help="Interval for logging temperature and margin updates",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

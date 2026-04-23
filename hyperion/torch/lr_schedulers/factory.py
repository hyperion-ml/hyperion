"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Union

import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .cos_lr import AdamCosineLR, CosineLR
from .exp_lr import ExponentialLR
from .invpow_lr import InvPowLR
from .lr_scheduler import LRScheduler
from .noam_lr import NoamLR
from .red_lr_on_plateau import ReduceLROnPlateau
from .triangular_lr import TriangularLR


class LRSchedulerFactory:
    """Factory for creating configured learning-rate schedulers."""

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        lrsch_type: str,
        decay_rate: float = 1 / 100,
        decay_steps: int = 100,
        power: float = 0.5,
        hold_steps: int = 10,
        t: int = 10,
        t_mul: int = 1,
        warm_restarts: bool = False,
        gamma: float = 1,
        monitor: str = "val_loss",
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        eps: float = 1e-8,
        min_lr: Union[float, Sequence[float]] = 0,
        warmup_steps: Optional[int] = None,
        d_model: Optional[int] = None,
        lr_factor: float = 1,
        update_lr_on_opt_step: bool = True,
    ) -> Optional[LRScheduler]:
        """Create a learning-rate scheduler instance.

        Args:
            optimizer: Wrapped optimizer.
            lrsch_type: Scheduler type identifier.
            decay_rate: Exponential decay factor.
            decay_steps: Number of steps associated with one exponential decay.
            power: Inverse-power decay exponent.
            hold_steps: Steps to hold initial LR before decay.
            t: Base cycle length for cyclic schedulers.
            t_mul: Cycle-length multiplier after each restart.
            warm_restarts: Enable warm restarts for cosine schedule.
            gamma: Max-LR multiplier after each restart.
            monitor: Metric key for plateau scheduler.
            mode: ``"min"`` or ``"max"`` for plateau scheduler.
            factor: LR reduction factor for plateau scheduler.
            patience: Patience (epochs) for plateau scheduler.
            threshold: Improvement threshold for plateau scheduler.
            threshold_mode: ``"rel"`` or ``"abs"`` threshold semantics.
            cooldown: Cooldown epochs for plateau scheduler.
            eps: Minimum effective LR change for plateau scheduler.
            min_lr: Scalar or per-group lower LR bound.
            warmup_steps: Linear warmup duration in optimizer steps. Uses
                scheduler-specific defaults when omitted.
            d_model: Transformer hidden size for Noam schedule.
            lr_factor: Scale factor for Noam schedule.
            update_lr_on_opt_step: Whether to update LR on optimizer steps.

        Returns:
            Scheduler instance, or ``None`` when ``lrsch_type == "none"``.

        Raises:
            ValueError: If ``lrsch_type`` is unknown.
        """

        if lrsch_type == "none":
            return None

        if lrsch_type == "exp_lr":
            return ExponentialLR(
                optimizer,
                decay_rate,
                decay_steps,
                hold_steps,
                min_lr=min_lr,
                warmup_steps=0 if warmup_steps is None else warmup_steps,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "invpow_lr":
            return InvPowLR(
                optimizer,
                power,
                hold_steps,
                min_lr=min_lr,
                warmup_steps=0 if warmup_steps is None else warmup_steps,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "noam_lr":
            return NoamLR(
                optimizer,
                d_model,
                lr_factor,
                min_lr=min_lr,
                warmup_steps=1 if warmup_steps is None else warmup_steps,
            )

        if lrsch_type == "cos_lr":
            return CosineLR(
                optimizer,
                t,
                t_mul,
                min_lr=min_lr,
                warmup_steps=0 if warmup_steps is None else warmup_steps,
                warm_restarts=warm_restarts,
                gamma=gamma,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type in ("triangular_lr", "triangular"):
            return TriangularLR(
                optimizer,
                t,
                t_mul,
                min_lr=min_lr,
                gamma=gamma,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "adamcos_lr":
            return AdamCosineLR(
                optimizer,
                t,
                t_mul,
                warmup_steps=0 if warmup_steps is None else warmup_steps,
                warm_restarts=warm_restarts,
                gamma=gamma,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "red_lr_on_plateau":
            return ReduceLROnPlateau(
                optimizer,
                monitor,
                mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                warmup_steps=0 if warmup_steps is None else warmup_steps,
                eps=eps,
            )

        raise ValueError(f"invalid lrsch_type={lrsch_type}")

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter a kwargs dictionary to args accepted by :meth:`create`."""
        valid_args = (
            "lrsch_type",
            "decay_rate",
            "decay_steps",
            "hold_steps",
            "power",
            "t",
            "t_mul",
            "warm_restarts",
            "gamma",
            "monitor",
            "mode",
            "factor",
            "patience",
            "threshold",
            "threshold_mode",
            "cooldown",
            "eps",
            "min_lr",
            "warmup_steps",
            "lr_factor",
            "d_model",
            "update_lr_on_opt_step",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Register LR scheduler CLI arguments in an argument parser."""
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--lrsch-type",
            type=str.lower,
            default="none",
            choices=[
                "none",
                "exp_lr",
                "invpow_lr",
                "cos_lr",
                "adamcos_lr",
                "red_lr_on_plateau",
                "noam_lr",
                "triangular_lr",
            ],
            help=(
                "Learning rate scheduler type (e.g., exp_lr, invpow_lr, cos_lr, "
                "adamcos_lr, red_lr_on_plateau, noam_lr, triangular_lr)."
            ),
        )

        parser.add_argument(
            "--decay-rate",
            default=1 / 100,
            type=float,
            help=("Exponential decay factor applied every decay_steps."),
        )
        parser.add_argument(
            "--decay-steps",
            default=100,
            type=int,
            help=("Number of steps between exponential decays."),
        )
        parser.add_argument(
            "--power",
            default=0.5,
            type=float,
            help=("Exponent for inverse power decay (lr ~ step^-power)."),
        )

        parser.add_argument(
            "--hold-steps",
            default=10,
            type=int,
            help=("Number of steps to hold the initial lr before decay."),
        )
        parser.add_argument(
            "--t",
            default=10,
            type=int,
            help=("Cycle length for cosine/triangular schedules (in steps)."),
        )
        parser.add_argument(
            "--t-mul",
            default=1,
            type=int,
            help=("Cycle length multiplier after each restart (cos/triangular)."),
        )
        parser.add_argument(
            "--gamma",
            default=1.0,
            type=float,
            help=("Max lr multiplier after each restart (cos/triangular)."),
        )

        parser.add_argument(
            "--warm-restarts",
            default=False,
            action=ActionYesNo,
            help=("Enable warm restarts in cosine schedules."),
        )

        parser.add_argument(
            "--monitor",
            default="val_loss",
            help=("Metric name to monitor for ReduceLROnPlateau."),
        )
        parser.add_argument(
            "--mode",
            default="min",
            choices=["min", "max"],
            help=("Whether lower or higher metric is better for plateau reduction."),
        )

        parser.add_argument(
            "--factor",
            default=0.1,
            type=float,
            help=("Multiply lr by this factor when plateau reduction triggers."),
        )

        parser.add_argument(
            "--patience",
            default=10,
            type=int,
            help=("Epochs with no improvement before reducing lr."),
        )

        parser.add_argument(
            "--threshold",
            default=1e-4,
            type=float,
            help=("Minimum change to qualify as an improvement."),
        )

        parser.add_argument(
            "--threshold_mode",
            default="rel",
            choices=["rel", "abs"],
            help=("Use relative or absolute threshold for improvements."),
        )

        parser.add_argument(
            "--cooldown",
            default=0,
            type=int,
            help=("Epochs to wait after a reduction before resuming checks."),
        )

        parser.add_argument(
            "--eps",
            default=1e-8,
            type=float,
            help=("Minimum lr change; smaller updates are ignored."),
        )

        parser.add_argument(
            "--min-lr", default=0, type=float, help=("Lower bound for learning rate.")
        )

        parser.add_argument(
            "--warmup-steps",
            default=None,
            type=int,
            help=("Steps to linearly warm up lr from 0 to the base value."),
        )

        parser.add_argument(
            "--d-model",
            default=None,
            type=int,
            help=("Transformer model dimension for Noam schedule."),
        )
        parser.add_argument(
            "--lr-factor",
            default=1,
            type=float,
            help=("Scale factor applied to the Noam learning rate."),
        )
        parser.add_argument(
            "--update-lr-on-opt-step",
            default=True,
            action=ActionYesNo,
            help=("Update lr per optimizer step instead of per epoch."),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='learning rate scheduler options')

    add_argparse_args = add_class_args

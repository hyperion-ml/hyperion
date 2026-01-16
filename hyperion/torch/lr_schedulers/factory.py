"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Sequence, Union

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
        warmup_steps: int = 0,
        d_model: Optional[int] = None,
        lr_factor: float = 1,
        update_lr_on_opt_step: bool = True,
    ) -> Optional[LRScheduler]:
        """Creates a  learning rate scheduler object.

        Args:
          optimizer: Pytorch optimizer object.
          lrsched_type: type of scheduler in ["none", "exp_lr", "invpow_lr",
                "cos_lr", "adamcos_lr", "red_lr_on_plateau", "noam_lr",
                        "triangular_lr"].
          decay_rate: the lr is multiplied by `decay_rate` after `decay_ste.ps`
          decay_steps: number of decay steps.
          power: the step/epoch number is ellebated to this power to compute the decay.
          hold_steps: number of steps until the lr starts decaying.
          t: period of the cycle.
          t_mul: period multiplier, after each cycle the period is multiplied by T_mul.
          warm_restarts: whether or not to do warm restarts.
          gamma: after each period, the maximum lr is multiplied by gamma, in cyclid schedulers.
          monitor: which metric to monitor in RedLROnPlateau scheduler.
          mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
          factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
          patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
          threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
          threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                 mode or best * ( 1 - threshold ) in `min` mode.
                 In `abs` mode, dynamic_threshold = best + threshold in
                 `max` mode or best - threshold in `min` mode. Default: 'rel'.
          cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
          eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.
                d_model: hidden dimension of transformer model.
          min_lr: minimum learning rate.
          warmup_steps: number of warm up steps to get the lr from 0 to the maximum lr.
          d_model: hidden dimension of transformer model.
          lr_factor: multiplies the Noam lr by this number.
          update_lr_on_opt_step: if True, updates the lr each time we update the model,
                otherwise after each epoch.
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
                warmup_steps=warmup_steps,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "invpow_lr":
            return InvPowLR(
                optimizer,
                power,
                hold_steps,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "noam_lr":
            return NoamLR(
                optimizer,
                d_model,
                lr_factor,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
            )

        if lrsch_type == "cos_lr":
            return CosineLR(
                optimizer,
                t,
                t_mul,
                min_lr=min_lr,
                warmup_steps=warmup_steps,
                warm_restarts=warm_restarts,
                gamma=gamma,
                update_lr_on_opt_step=update_lr_on_opt_step,
            )

        if lrsch_type == "triangular":
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
                warmup_steps=warmup_steps,
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
                warmup_steps=warmup_steps,
                eps=eps,
            )

        raise ValueError(f"invalid lrsch_type={lrsch_type}")

    @staticmethod
    def filter_args(**kwargs):
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
    def add_class_args(parser, prefix=None):
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
            default=0,
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

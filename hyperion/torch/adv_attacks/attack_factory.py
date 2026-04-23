"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from .adv_attack import AdvAttack
from .carlini_wagner_l0 import CarliniWagnerL0
from .carlini_wagner_l2 import CarliniWagnerL2
from .carlini_wagner_linf import CarliniWagnerLInf
from .fgsm_attack import FGSMAttack
from .iter_fgsm_attack import IterFGSMAttack
from .pgd_attack import PGDAttack
from .rand_fgsm_attack import RandFGSMAttack
from .snr_fgsm_attack import SNRFGSMAttack


class AttackFactory:
    """Factory for project-native adversarial attacks."""

    @staticmethod
    def create(
        model: nn.Module,
        attack_type: str,
        eps: float = 0.1,
        snr: float = 100,
        alpha: float = 0.01,
        norm: float = float("inf"),
        random_eps: bool = False,
        num_random_init: int = 0,
        confidence: float = 0.0,
        lr: float = 1e-2,
        binary_search_steps: int = 9,
        max_iter: int = 10,
        abort_early: bool = True,
        initial_c: float = 1e-3,
        reduce_c: bool = False,
        c_incr_factor: float = 2,
        tau_decr_factor: float = 0.9,
        indep_channels: bool = False,
        norm_time: bool = False,
        time_dim: Optional[int] = None,
        use_snr: bool = False,
        loss: Optional[nn.Module] = None,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        eps_scale: float = 1,
    ) -> AdvAttack:
        """Create an adversarial attack instance.

        Args:
          model: Model under attack.
          attack_type: Attack identifier.
          eps: Perturbation budget.
          snr: Target SNR for SNR-based FGSM.
          alpha: Step size for iterative/randomized attacks.
          norm: Threat-model norm.
          random_eps: Whether to randomize epsilon (PGD).
          num_random_init: Number of random initializations.
          confidence: Confidence parameter for Carlini-Wagner attacks.
          lr: Optimizer learning rate for iterative attacks.
          binary_search_steps: Binary search steps for CW-L2.
          max_iter: Maximum iterations for iterative attacks.
          abort_early: Whether to abort optimization early.
          initial_c: Initial ``c`` for Carlini-Wagner variants.
          reduce_c: Whether to reduce ``c`` in CW-L0/Linf.
          c_incr_factor: Multiplicative increase for ``c``.
          tau_decr_factor: Multiplicative decrease for ``tau`` in CW-Linf.
          indep_channels: Use independent channels in CW-L0.
          norm_time: Whether to normalize norms by time length.
          time_dim: Time axis used by ``norm_time``.
          use_snr: Whether to use SNR objective in CW-L2.
          loss: Optional custom loss module.
          targeted: Whether the attack is targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.
          eps_scale: Global scale factor applied to ``eps``/``alpha``.

        Returns:
          Configured attack instance.
        """

        eps = eps * eps_scale
        alpha = alpha * eps_scale
        norm = float(norm)

        if attack_type in ("fgsm", "iter-fgsm", "rand-fgsm", "pgd") and eps <= 0:
            raise ValueError(f"{attack_type} requires eps > 0, got eps={eps}")

        if attack_type in ("iter-fgsm", "rand-fgsm", "pgd"):
            if alpha <= 0:
                raise ValueError(f"{attack_type} requires alpha > 0, got alpha={alpha}")
            if alpha >= eps:
                raise ValueError(
                    f"{attack_type} requires alpha < eps, got alpha={alpha}, eps={eps}"
                )

        if attack_type == "fgsm":
            return FGSMAttack(
                model,
                eps,
                loss=loss,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "snr-fgsm":
            return SNRFGSMAttack(
                model,
                snr,
                loss=loss,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "rand-fgsm":
            return RandFGSMAttack(
                model,
                eps,
                alpha,
                loss=loss,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "iter-fgsm":
            return IterFGSMAttack(
                model,
                eps,
                alpha,
                loss=loss,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "cw-l2":
            return CarliniWagnerL2(
                model,
                confidence,
                lr,
                binary_search_steps,
                max_iter,
                abort_early,
                initial_c,
                norm_time=norm_time,
                time_dim=time_dim,
                use_snr=use_snr,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "cw-l0":
            return CarliniWagnerL0(
                model,
                confidence,
                lr,
                max_iter,
                abort_early,
                initial_c,
                reduce_c,
                c_incr_factor,
                indep_channels,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "cw-linf":
            return CarliniWagnerLInf(
                model,
                confidence,
                lr,
                max_iter,
                abort_early,
                initial_c,
                reduce_c,
                c_incr_factor,
                tau_decr_factor,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        if attack_type == "pgd":
            return PGDAttack(
                model,
                eps,
                alpha,
                norm,
                max_iter,
                random_eps,
                num_random_init,
                loss=loss,
                norm_time=norm_time,
                time_dim=time_dim,
                targeted=targeted,
                range_min=range_min,
                range_max=range_max,
            )

        raise Exception("%s is not a valid attack type" % (attack_type))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter and normalize keyword arguments for :meth:`create`.

        Args:
          **kwargs: Unstructured attack options.

        Returns:
          Filtered dictionary accepted by :meth:`create`.
        """
        if "no_abort" in kwargs:
            kwargs["abort_early"] = not kwargs["no_abort"]

        if "norm" in kwargs:
            if isinstance(kwargs["norm"], str):
                kwargs["norm"] = float(kwargs["norm"])

        return filter_func_args(AttackFactory.create, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Register CLI arguments for attack creation.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional nested prefix for grouped arguments.

        Returns:
          None.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--attack-type",
            type=str.lower,
            default="fgsm",
            choices=[
                "fgsm",
                "snr-fgsm",
                "rand-fgsm",
                "iter-fgsm",
                "cw-l0",
                "cw-l2",
                "cw-linf",
                "pgd",
            ],
            help=("Attack type"),
        )

        parser.add_argument(
            "--norm",
            default="inf",
            choices=["inf", "1", "2"],
            help=("Attack perturbation norm"),
        )

        parser.add_argument(
            "--eps",
            default=0.1,
            type=float,
            help=("attack epsilon, upper bound for the perturbation norm"),
        )

        parser.add_argument(
            "--snr",
            default=100,
            type=float,
            help=(
                "upper bound for the signal-to-noise ratio of " "the perturved signal"
            ),
        )

        parser.add_argument(
            "--alpha",
            default=0.01,
            type=float,
            help=("alpha for iter and rand fgsm attack"),
        )

        parser.add_argument(
            "--random-eps",
            default=False,
            action="store_true",
            help=("use random epsilon in PGD attack"),
        )

        parser.add_argument(
            "--confidence",
            default=0,
            type=float,
            help=("confidence for carlini-wagner attack"),
        )

        parser.add_argument(
            "--lr",
            default=1e-2,
            type=float,
            help=("learning rate for attack optimizers"),
        )

        parser.add_argument(
            "--binary-search-steps",
            default=9,
            type=int,
            help=("num bin. search steps in carlini-wagner-l2 attack"),
        )

        parser.add_argument(
            "--max-iter",
            default=10,
            type=int,
            help=("max. num. of optim iters in attack"),
        )

        parser.add_argument(
            "--initial-c",
            default=1e-2,
            type=float,
            help=(
                "initial weight of constraint function f in " "carlini-wagner attack"
            ),
        )

        parser.add_argument(
            "--reduce-c",
            default=False,
            action="store_true",
            help=("allow to reduce c in carline-wagner-l0/inf attack"),
        )

        parser.add_argument(
            "--c-incr-factor",
            default=2,
            type=float,
            help=("factor to increment c in carline-wagner-l0/inf attack"),
        )

        parser.add_argument(
            "--tau-decr-factor",
            default=0.75,
            type=float,
            help=("factor to reduce tau in carline-wagner-linf attack"),
        )

        parser.add_argument(
            "--indep-channels",
            default=False,
            action="store_true",
            help=("consider independent input channels in " "carline-wagner-l0 attack"),
        )

        parser.add_argument(
            "--no-abort",
            default=False,
            action="store_true",
            help=("do not abort early in optimizer iterations"),
        )

        parser.add_argument(
            "--num-random-init",
            default=0,
            type=int,
            help=("number of random initializations in PGD attack"),
        )

        parser.add_argument(
            "--targeted",
            default=False,
            action="store_true",
            help="use targeted attack intead of non-targeted",
        )

        parser.add_argument(
            "--use-snr",
            default=False,
            action="store_true",
            help=(
                "In carlini-wagner attack maximize SNR instead of "
                "minimize perturbation norm"
            ),
        )

        parser.add_argument(
            "--norm-time",
            default=False,
            action="store_true",
            help=("normalize norm by number of samples in time dimension"),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='adversarial attack options')

    add_argparse_args = add_class_args

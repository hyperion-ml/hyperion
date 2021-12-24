"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

from .fgsm_attack import FGSMAttack
from .snr_fgsm_attack import SNRFGSMAttack
from .rand_fgsm_attack import RandFGSMAttack
from .iter_fgsm_attack import IterFGSMAttack
from .carlini_wagner_l2 import CarliniWagnerL2
from .carlini_wagner_l0 import CarliniWagnerL0
from .carlini_wagner_linf import CarliniWagnerLInf
from .pgd_attack import PGDAttack


class AttackFactory(object):
    @staticmethod
    def create(
        model,
        attack_type,
        eps=0,
        snr=100,
        alpha=0,
        norm=float("inf"),
        random_eps=False,
        num_random_init=0,
        confidence=0.0,
        lr=1e-2,
        binary_search_steps=9,
        max_iter=10,
        abort_early=True,
        c=1e-3,
        reduce_c=False,
        c_incr_factor=2,
        tau_decr_factor=0.9,
        indep_channels=False,
        norm_time=False,
        time_dim=None,
        use_snr=False,
        loss=None,
        targeted=False,
        range_min=None,
        range_max=None,
        eps_scale=1,
    ):

        eps = eps * eps_scale
        alpha = alpha * eps_scale

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
                c,
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
                c,
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
                c,
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
    def filter_args(**kwargs):
        if "no_abort" in kwargs:
            kwargs["abort_early"] = not kwargs["no_abort"]

        if "norm" in kwargs:
            if isinstance(kwargs["norm"], str):
                kwargs["norm"] = float(kwargs["norm"])

        valid_args = (
            "attack_type",
            "eps",
            "snr",
            "norm",
            "random_eps",
            "num_random_init",
            "alpha",
            "confidence",
            "lr",
            "binary_search_steps",
            "max_iter",
            "abort_early",
            "c",
            "reduce_c",
            "c_incr_factor",
            "tau_decr_factor",
            "indep_channels",
            "use_snr",
            "norm_time",
            "targeted",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
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
            type=float,
            default=float("inf"),
            choices=[float("inf"), 1, 2],
            help=("Attack perturbation norm"),
        )

        parser.add_argument(
            "--eps",
            default=0,
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
            default=0,
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
            "--c",
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

"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Iterable, Optional, Type, Union

import torch
import torch.optim as optim
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_args, filter_func_args


class OptimizerFactory:
    """Build torch optimizers from a normalized set of keyword arguments.

    The factory accepts a superset of optimizer hyperparameters and forwards only
    the ones used by the selected optimizer type.
    """

    @staticmethod
    def create(
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        opt_type: str,
        lr: float,
        momentum: float = 0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        nesterov: bool = False,
        lambd: float = 0.0001,
        asgd_alpha: float = 0.75,
        t0: float = 1000000.0,
        rmsprop_alpha: float = 0.99,
        centered: bool = False,
        lr_decay: float = 0,
        init_acc_val: float = 0,
        max_iter: int = 20,
    ) -> optim.Optimizer:
        """Create an optimizer instance.

        Args:
            params: Iterable of model parameters or parameter-group dicts.
            opt_type: Optimizer name (e.g., ``"adam"``, ``"sgd"``, ``"radam"``).
            lr: Learning rate.
            momentum: SGD/RMSProp momentum.
            beta1: First moment coefficient for Adam-family optimizers.
            beta2: Second moment coefficient for Adam-family optimizers.
            rho: Adadelta decay factor.
            eps: Numerical-stability constant for adaptive optimizers.
            weight_decay: L2 weight decay.
            amsgrad: Use AMSGrad variant for Adam/AdamW.
            nesterov: Enable Nesterov momentum for SGD.
            lambd: ASGD decay term.
            asgd_alpha: ASGD power for eta update.
            t0: ASGD averaging start point.
            rmsprop_alpha: RMSProp smoothing constant.
            centered: Use centered RMSProp.
            lr_decay: AdaGrad learning-rate decay.
            init_acc_val: AdaGrad initial accumulator value.
            max_iter: Maximum iterations for LBFGS.

        Returns:
            Instantiated ``torch.optim.Optimizer``.

        Raises:
            ValueError: If ``opt_type`` is not recognized.
        """
        kwargs = locals()
        base_opt: Optional[Type[optim.Optimizer]] = None
        if opt_type == "sgd":
            valid_args = ("lr", "momentum", "weight_decay", "nesterov")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["dampening"] = 0
            base_opt = optim.SGD
            # return optim.SGD(params, lr, momentum=momentum, dampening=0,
            #                  weight_decay=weight_decay, nesterov=nesterov)

        elif opt_type == "adam":
            betas = (beta1, beta2)
            valid_args = ("lr", "eps", "weight_decay", "amsgrad")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["betas"] = betas
            base_opt = optim.Adam

        elif opt_type == "adamw":
            betas = (beta1, beta2)
            valid_args = ("lr", "eps", "weight_decay", "amsgrad")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["betas"] = betas
            base_opt = optim.AdamW

        elif opt_type == "radam":
            betas = (beta1, beta2)
            valid_args = ("lr", "eps", "weight_decay")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["betas"] = betas
            base_opt = optim.RAdam

        elif opt_type == "adadelta":
            valid_args = ("lr", "eps", "weight_decay", "rho")
            opt_args = filter_args(valid_args, kwargs)
            base_opt = optim.Adadelta

        elif opt_type == "adagrad":
            valid_args = ("lr", "lr_decay", "weight_decay")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["initial_accumulator_value"] = init_acc_val
            base_opt = optim.Adagrad

        elif opt_type == "sparse_adam":
            betas = (beta1, beta2)
            valid_args = ("lr", "eps")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["betas"] = betas
            base_opt = optim.SparseAdam

        elif opt_type == "adamax":
            betas = (beta1, beta2)
            valid_args = ("lr", "eps", "weight_decay")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["betas"] = betas
            base_opt = optim.Adamax

        elif opt_type == "asgd":
            valid_args = ("lr", "lambd", "t0", "weight_decay")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["alpha"] = asgd_alpha
            base_opt = optim.ASGD
            # return optim.ASGD(params, lr, lambd=lambd, alpha=asgd_alpha, t0=t0,
            #                   weight_decay=weight_decay)

        elif opt_type == "lbfgs":
            valid_args = ("lr", "max_iter")
            opt_args = filter_args(valid_args, kwargs)
            base_opt = optim.LBFGS

        elif opt_type == "rmsprop":
            valid_args = ("lr", "eps", "momentum", "weight_decay", "centered")
            opt_args = filter_args(valid_args, kwargs)
            opt_args["alpha"] = rmsprop_alpha
            base_opt = optim.RMSprop

        elif opt_type == "rprop":
            opt_args = {"lr": lr, "etas": (0.5, 1.2), "step_sizes": (1e-06, 50)}
            base_opt = optim.Rprop

        if base_opt is None:
            raise ValueError("unknown optimizer %s" % opt_type)

        return base_opt(params, **opt_args)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter a raw kwargs dictionary to arguments accepted by ``create``."""
        return filter_func_args(OptimizerFactory.create, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register optimizer configuration arguments in a parser.

        Args:
            parser: Destination argument parser.
            prefix: Optional nested prefix used to add optimizer args as a group.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--opt-type",
            type=str.lower,
            default="adam",
            choices=[
                "sgd",
                "adam",
                "adamw",
                "radam",
                "adadelta",
                "adagrad",
                "sparse_adam",
                "adamax",
                "asgd",
                "lbfgs",
                "rmsprop",
                "rprop",
            ],
            help=(
                "Optimizers: SGD, Adam, AdaDelta, AdaGrad, SparseAdam "
                "AdaMax, ASGD, LBFGS, RMSprop, Rprop"
            ),
        )
        parser.add_argument(
            "--lr", default=0.001, type=float, help=("Initial learning rate")
        )
        parser.add_argument("--momentum", default=0.6, type=float, help=("Momentum"))
        parser.add_argument(
            "--beta1",
            default=0.9,
            type=float,
            help=(
                "Beta_1 in Adam optimizers,  "
                "coefficient used for computing "
                "running averages of gradient"
            ),
        )
        parser.add_argument(
            "--beta2",
            default=0.99,
            type=float,
            help=(
                "Beta_2 in Adam optimizers"
                "coefficient used for computing "
                "running averages of gradient square"
            ),
        )
        parser.add_argument(
            "--rho",
            default=0.9,
            type=float,
            help=(
                "Rho in AdaDelta,"
                "coefficient used for computing a "
                "running average of squared gradients"
            ),
        )
        parser.add_argument(
            "--eps",
            default=1e-8,
            type=float,
            help=(
                "Epsilon in RMSprop and Adam optimizers "
                "term added to the denominator "
                "to improve numerical stability"
            ),
        )

        parser.add_argument(
            "--weight-decay",
            default=1e-6,
            type=float,
            help=("L2 regularization coefficient"),
        )

        parser.add_argument(
            "--amsgrad",
            default=False,
            action="store_true",
            help=("AMSGrad variant of Adam"),
        )

        parser.add_argument(
            "--nesterov",
            default=False,
            action="store_true",
            help=("Use Nesterov momentum in SGD"),
        )

        parser.add_argument(
            "--lambd", default=0.0001, type=float, help=("decay term in ASGD")
        )

        parser.add_argument(
            "--asgd-alpha",
            default=0.75,
            type=float,
            help=("power for eta update in ASGD"),
        )

        parser.add_argument(
            "--t0",
            default=1e6,
            type=float,
            help=("point at which to start averaging in ASGD"),
        )

        parser.add_argument(
            "--rmsprop-alpha",
            default=0.99,
            type=float,
            help=("smoothing constant in RMSprop"),
        )

        parser.add_argument(
            "--centered",
            default=False,
            action="store_true",
            help=("Compute centered RMSprop, gradient normalized " "by its variance"),
        )

        parser.add_argument(
            "--lr-decay",
            default=1e-6,
            type=float,
            help=("Learning rate decay in AdaGrad optimizer"),
        )

        parser.add_argument(
            "--init-acc-val",
            default=0,
            type=float,
            help=("Init accum value in Adagrad"),
        )

        parser.add_argument(
            "--max-iter", default=20, type=int, help=("max iterations in LBFGS")
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

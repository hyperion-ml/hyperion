"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import torch
from jsonargparse import ActionParser, ArgumentParser

from ...utils.misc import filter_func_args
from .adv_attack import AdvAttack
from .attack_factory import AttackFactory as AF


class RandomAttackFactory:
    """Sampler that draws random attack configurations.

    Attributes:
      attack_types: Candidate attack names to sample from.
      norms: Candidate threat-model norms for norm-based attacks.
      random_eps: Whether PGD epsilon should be randomized.
      targeted: Whether sampled attacks are targeted.
      range_min: Optional minimum clamp value for sampled attacks.
      range_max: Optional maximum clamp value for sampled attacks.
      eps_scale: Global epsilon scaling factor.
    """

    def __init__(
        self,
        attack_types: Sequence[str],
        min_eps: float = 1e-5,
        max_eps: float = 0.1,
        min_snr: float = 30,
        max_snr: float = 60,
        min_alpha: float = 1e-5,
        max_alpha: float = 0.02,
        norms: Optional[Sequence[float]] = None,
        random_eps: bool = False,
        min_num_random_init: int = 0,
        max_num_random_init: int = 3,
        min_confidence: float = 0,
        max_confidence: float = 1,
        min_lr: float = 1e-3,
        max_lr: float = 1e-2,
        min_binary_search_steps: int = 9,
        max_binary_search_steps: int = 9,
        min_iter: int = 5,
        max_iter: int = 10,
        abort_early: bool = True,
        min_c: float = 1e-3,
        max_c: float = 1e-2,
        reduce_c: bool = False,
        c_incr_factor: float = 2,
        tau_decr_factor: float = 0.9,
        indep_channels: bool = False,
        norm_time: bool = False,
        time_dim: Optional[int] = None,
        use_snr: bool = False,
        loss: Optional[torch.nn.Module] = None,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        eps_scale: float = 1,
    ) -> None:
        """Initialize random attack sampler.

        Args:
          attack_types: Candidate attack names.
          min_eps: Minimum epsilon for log-uniform sampling.
          max_eps: Maximum epsilon for log-uniform sampling.
          min_snr: Minimum SNR for uniform sampling.
          max_snr: Maximum SNR for uniform sampling.
          min_alpha: Minimum alpha for log-uniform sampling.
          max_alpha: Maximum alpha for log-uniform sampling.
          norms: Candidate norm values (defaults to ``[inf]``).
          random_eps: Whether PGD uses random epsilon.
          min_num_random_init: Minimum random restarts for PGD.
          max_num_random_init: Maximum random restarts for PGD.
          min_confidence: Minimum confidence for CW attacks.
          max_confidence: Maximum confidence for CW attacks.
          min_lr: Minimum optimizer learning rate.
          max_lr: Maximum optimizer learning rate.
          min_binary_search_steps: Minimum binary-search steps for CW-L2.
          max_binary_search_steps: Maximum binary-search steps for CW-L2.
          min_iter: Minimum iterative steps.
          max_iter: Maximum iterative steps.
          abort_early: Whether sampled attacks abort early when supported.
          min_c: Minimum initial ``c`` for CW attacks.
          max_c: Maximum initial ``c`` for CW attacks.
          reduce_c: Whether sampled CW-L0/Linf attacks reduce ``c``.
          c_incr_factor: ``c`` increase factor.
          tau_decr_factor: Tau decrease factor for CW-Linf.
          indep_channels: Independent-channel mode for CW-L0.
          norm_time: Whether sampled attacks normalize norms by time.
          time_dim: Time axis used with ``norm_time``.
          use_snr: Whether sampled CW-L2 attacks use SNR objective.
          loss: Optional custom loss module.
          targeted: Whether sampled attacks are targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.
          eps_scale: Global epsilon scaling factor.

        Returns:
          None.
        """

        if len(attack_types) == 0:
            raise ValueError("RandomAttackFactory requires at least one attack_type")
        if min_eps <= 0 or max_eps <= 0 or min_eps > max_eps:
            raise ValueError(
                f"RandomAttackFactory requires 0 < min_eps <= max_eps, got min_eps={min_eps}, max_eps={max_eps}"
            )
        if min_alpha <= 0 or max_alpha <= 0 or min_alpha > max_alpha:
            raise ValueError(
                "RandomAttackFactory requires 0 < min_alpha <= max_alpha, "
                f"got min_alpha={min_alpha}, max_alpha={max_alpha}"
            )
        if min_snr > max_snr:
            raise ValueError(
                f"RandomAttackFactory requires min_snr <= max_snr, got min_snr={min_snr}, max_snr={max_snr}"
            )
        if min_num_random_init < 0 or max_num_random_init < 0:
            raise ValueError(
                "RandomAttackFactory requires non-negative random init bounds, "
                f"got min_num_random_init={min_num_random_init}, max_num_random_init={max_num_random_init}"
            )
        if min_num_random_init > max_num_random_init:
            raise ValueError(
                "RandomAttackFactory requires min_num_random_init <= max_num_random_init, "
                f"got min_num_random_init={min_num_random_init}, max_num_random_init={max_num_random_init}"
            )
        if min_binary_search_steps < 1 or max_binary_search_steps < 1:
            raise ValueError(
                "RandomAttackFactory requires binary search steps >= 1, "
                f"got min_binary_search_steps={min_binary_search_steps}, "
                f"max_binary_search_steps={max_binary_search_steps}"
            )
        if min_binary_search_steps > max_binary_search_steps:
            raise ValueError(
                "RandomAttackFactory requires min_binary_search_steps <= max_binary_search_steps, "
                f"got min_binary_search_steps={min_binary_search_steps}, "
                f"max_binary_search_steps={max_binary_search_steps}"
            )
        if min_iter < 1 or max_iter < 1 or min_iter > max_iter:
            raise ValueError(
                f"RandomAttackFactory requires 1 <= min_iter <= max_iter, got min_iter={min_iter}, max_iter={max_iter}"
            )
        if min_lr <= 0 or max_lr <= 0 or min_lr > max_lr:
            raise ValueError(
                f"RandomAttackFactory requires 0 < min_lr <= max_lr, got min_lr={min_lr}, max_lr={max_lr}"
            )
        if min_c <= 0 or max_c <= 0 or min_c > max_c:
            raise ValueError(
                f"RandomAttackFactory requires 0 < min_c <= max_c, got min_c={min_c}, max_c={max_c}"
            )

        norms_list = [float("inf")] if norms is None else [float(n) for n in norms]
        if len(norms_list) == 0:
            raise ValueError("RandomAttackFactory requires at least one norm")
        valid_norms = {1.0, 2.0}
        if not all(math.isinf(n) or n in valid_norms for n in norms_list):
            raise ValueError(
                f"RandomAttackFactory only supports norms in {{1, 2, inf}}, got norms={norms_list}"
            )

        self.attack_types = attack_types
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.norms = norms_list
        self.random_eps = random_eps
        self.min_num_random_init = min_num_random_init
        self.max_num_random_init = max_num_random_init
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_binary_search_steps = min_binary_search_steps
        self.max_binary_search_steps = max_binary_search_steps
        self.abort_early = abort_early
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.min_c = min_c
        self.max_c = max_c
        self.reduce_c = reduce_c
        self.c_incr_factor = c_incr_factor
        self.tau_decr_factor = tau_decr_factor
        self.indep_channels = indep_channels
        self.norm_time = norm_time
        self.time_dim = time_dim
        self.use_snr = use_snr
        self.loss = loss
        self.targeted = targeted
        self.range_min = range_min
        self.range_max = range_max
        self.eps_scale = eps_scale

    @staticmethod
    def _choice(n: int) -> int:
        """Sample a random index in ``[0, n)``.

        Args:
          n: Upper bound (exclusive).

        Returns:
          Random integer index.
        """
        return torch.randint(low=0, high=n, size=(1,)).item()

    @staticmethod
    def _randint(min_val: int, max_val: int) -> int:
        """Sample an integer uniformly in ``[min_val, max_val]``.

        Args:
          min_val: Minimum value (inclusive).
          max_val: Maximum value (inclusive).

        Returns:
          Random integer.
        """
        return torch.randint(low=min_val, high=max_val + 1, size=(1,)).item()

    @staticmethod
    def _uniform(min_val: float, max_val: float) -> float:
        """Sample a float uniformly in ``[min_val, max_val]``.

        Args:
          min_val: Minimum value.
          max_val: Maximum value.

        Returns:
          Random float.
        """
        return (max_val - min_val) * torch.rand(size=(1,)).item() + min_val

    @staticmethod
    def _log_uniform(min_val: float, max_val: float) -> float:
        """Sample a float log-uniformly in ``[min_val, max_val]``.

        Args:
          min_val: Minimum value.
          max_val: Maximum value.

        Returns:
          Random float.
        """
        log_x = (math.log(max_val) - math.log(min_val)) * torch.rand(
            size=(1,)
        ).item() + math.log(min_val)
        return math.exp(log_x)

    def _sample_attack_args(self) -> Dict[str, Any]:
        """Sample one attack configuration dictionary.

        Args:
          None.

        Returns:
          Dictionary of attack arguments accepted by :class:`AttackFactory`.
        """
        attack_args: Dict[str, Any] = {}
        attack_idx = self._choice(len(self.attack_types))
        attack_args["attack_type"] = self.attack_types[attack_idx]
        eps = self._log_uniform(self.min_eps, self.max_eps)
        attack_args["eps"] = eps
        attack_args["snr"] = self._uniform(self.min_snr, self.max_snr)
        max_alpha = min(0.999 * eps, self.max_alpha)
        min_alpha = min(self.min_alpha, max_alpha)
        attack_args["alpha"] = self._log_uniform(min_alpha, max_alpha)
        attack_args["norm"] = self.norms[self._choice(len(self.norms))]
        attack_args["random_eps"] = self.random_eps
        attack_args["num_random_init"] = self._randint(
            self.min_num_random_init, self.max_num_random_init
        )
        attack_args["confidence"] = self._uniform(
            self.min_confidence, self.max_confidence
        )
        attack_args["lr"] = self._uniform(self.min_lr, self.max_lr)
        attack_args["binary_search_steps"] = self._randint(
            self.min_binary_search_steps, self.max_binary_search_steps
        )
        attack_args["max_iter"] = self._randint(self.min_iter, self.max_iter)
        attack_args["abort_early"] = self.abort_early
        attack_args["initial_c"] = self._uniform(self.min_c, self.max_c)
        attack_args["reduce_c"] = self.reduce_c
        attack_args["c_incr_factor"] = self.c_incr_factor
        attack_args["tau_decr_factor"] = self.tau_decr_factor
        attack_args["indep_channels"] = self.indep_channels
        attack_args["norm_time"] = self.norm_time
        attack_args["time_dim"] = self.time_dim
        attack_args["use_snr"] = self.use_snr
        attack_args["targeted"] = self.targeted
        attack_args["range_min"] = self.range_min
        attack_args["range_max"] = self.range_max
        attack_args["eps_scale"] = self.eps_scale
        attack_args["loss"] = self.loss

        return attack_args

    def sample_attack(self, model: Optional[torch.nn.Module] = None) -> AdvAttack:
        """Create one sampled attack instance.

        Args:
          model: Model under attack.

        Returns:
          Instantiated attack object.
        """
        attack_args = self._sample_attack_args()
        attack_args["model"] = model
        return AF.create(**attack_args)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter and normalize keyword arguments for constructor use.

        Args:
          **kwargs: Unstructured sampler options.

        Returns:
          Filtered dictionary accepted by :class:`RandomAttackFactory`.
        """

        if "no_abort" in kwargs:
            kwargs["abort_early"] = not kwargs["no_abort"]

        if "norms" in kwargs:
            kwargs["norms"] = [float(a) for a in kwargs["norms"]]

        return filter_func_args(RandomAttackFactory.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None
    ) -> None:
        """Register CLI arguments for random attack sampling.

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
            "--attack-types",
            type=str.lower,
            default=["fgsm"],
            nargs="+",
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
            help=("Attack types"),
        )

        parser.add_argument(
            "--norms",
            default=["inf"],
            nargs="+",
            choices=["inf", "1", "2"],
            help=("Attack perturbation norms"),
        )

        parser.add_argument(
            "--min-eps",
            default=1e-5,
            type=float,
            help=("attack min epsilon, upper bound for the perturbation norm"),
        )

        parser.add_argument(
            "--max-eps",
            default=0.1,
            type=float,
            help=("attack max epsilon, upper bound for the perturbation norm"),
        )

        parser.add_argument(
            "--min-snr",
            default=30,
            type=float,
            help=(
                "min upper bound for the signal-to-noise ratio of the "
                "perturbed signal"
            ),
        )

        parser.add_argument(
            "--max-snr",
            default=60,
            type=float,
            help=(
                "max upper bound for the signal-to-noise ratio of the "
                "perturbed signal"
            ),
        )

        parser.add_argument(
            "--min-alpha",
            default=1e-5,
            type=float,
            help=("min alpha for iter and rand fgsm attack"),
        )

        parser.add_argument(
            "--max-alpha",
            default=0.02,
            type=float,
            help=("max alpha for iter and rand fgsm attack"),
        )

        parser.add_argument(
            "--random-eps",
            default=False,
            action="store_true",
            help=("use random epsilon in PGD attack"),
        )

        parser.add_argument(
            "--min-confidence",
            default=0,
            type=float,
            help=("min confidence for carlini-wagner attack"),
        )

        parser.add_argument(
            "--max-confidence",
            default=1,
            type=float,
            help=("max confidence for carlini-wagner attack"),
        )

        parser.add_argument(
            "--min-lr",
            default=1e-3,
            type=float,
            help=("min learning rate for attack optimizers"),
        )

        parser.add_argument(
            "--max-lr",
            default=1e-2,
            type=float,
            help=("max learning rate for attack optimizers"),
        )

        parser.add_argument(
            "--min-binary-search-steps",
            default=9,
            type=int,
            help=("min num bin. search steps in carlini-wagner-l2 attack"),
        )

        parser.add_argument(
            "--max-binary-search-steps",
            default=9,
            type=int,
            help=("max num bin. search steps in carlini-wagner-l2 attack"),
        )

        parser.add_argument(
            "--min-iter",
            default=5,
            type=int,
            help=("min maximum. num. of optim iters in attack"),
        )

        parser.add_argument(
            "--max-iter",
            default=10,
            type=int,
            help=("max maximum num. of optim iters in attack"),
        )

        parser.add_argument(
            "--min-c",
            default=1e-3,
            type=float,
            help=(
                "min initial weight of constraint function f "
                "in carlini-wagner attack"
            ),
        )

        parser.add_argument(
            "--max-c",
            default=1e-2,
            type=float,
            help=(
                "max initial weight of constraint function f "
                "in carlini-wagner attack"
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
            help=("consider independent input channels in " "carlini-wagner-l0 attack"),
        )

        parser.add_argument(
            "--no-abort",
            default=False,
            action="store_true",
            help=("do not abort early in optimizer iterations"),
        )

        parser.add_argument(
            "--min-num-random-init",
            default=1,
            type=int,
            help=("min number of random initializations in PGD attack"),
        )

        parser.add_argument(
            "--max-num-random-init",
            default=5,
            type=int,
            help=("max number of random initializations in PGD attack"),
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

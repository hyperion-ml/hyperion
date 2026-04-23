"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .adv_attack import AdvAttack


class CarliniWagner(AdvAttack):
    """Base class for Carlini-Wagner attacks.

    Attributes:
      confidence: Confidence margin for the attack objective.
      lr: Optimizer learning rate.
      max_iter: Maximum optimization steps.
      abort_early: Whether to stop early once successful.
      initial_c: Initial weight for the classification term.
      is_binary: Cached flag indicating binary-vs-multiclass output.
      box_scale: Affine scaling factor for tanh-space conversion.
      box_bias: Affine bias for tanh-space conversion.
      norm_time: Whether norm terms are time-normalized.
      time_dim: Time axis index when ``norm_time`` is used.
      use_snr: Whether to optimize SNR-derived objective when supported.
    """

    def __init__(
        self,
        model: nn.Module,
        confidence: float = 0.0,
        lr: float = 1e-2,
        max_iter: int = 10000,
        abort_early: bool = True,
        initial_c: float = 1e-3,
        norm_time: bool = False,
        time_dim: Optional[int] = None,
        use_snr: bool = False,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> None:
        """Initialize Carlini-Wagner attack base.

        Args:
          model: Model under attack.
          confidence: Confidence margin in the objective.
          lr: Optimizer learning rate.
          max_iter: Maximum optimization steps.
          abort_early: Whether to stop optimization early.
          initial_c: Initial positive weight for the classification term.
          norm_time: Whether to normalize norms by time length.
          time_dim: Time-axis index used by ``norm_time``.
          use_snr: Whether to optimize SNR-based distance.
          targeted: Whether the attack is targeted.
          range_min: Minimum clamp value (required for CW attacks).
          range_max: Maximum clamp value (required for CW attacks).

        Returns:
          None.
        """

        super().__init__(model, None, targeted, range_min, range_max)
        if self.range_min is None or self.range_max is None:
            raise ValueError(
                "Carlini-Wagner attacks require non-None range_min and range_max"
            )
        if initial_c <= 0:
            raise ValueError(
                f"Carlini-Wagner attacks require initial_c > 0, got initial_c={initial_c}"
            )
        if norm_time and time_dim is None:
            raise ValueError("Carlini-Wagner attacks require time_dim when norm_time=True")
        self.confidence = confidence
        self.lr = lr
        self.max_iter = max_iter
        self.abort_early = abort_early
        self.initial_c = initial_c
        self.is_binary = None
        self.box_scale = (self.range_max - self.range_min) / 2
        self.box_bias = (self.range_max + self.range_min) / 2
        self.norm_time = norm_time
        self.time_dim = time_dim
        self.use_snr = use_snr

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return attack metadata.

        Args:
          None.

        Returns:
          Dictionary with Carlini-Wagner configuration fields.
        """
        info = super().attack_info
        new_info = {
            "confidence": self.confidence,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "abort_early": self.abort_early,
            "initial_c": self.initial_c,
            "norm_time": self.norm_time,
            "use_snr": self.use_snr,
        }
        info.update(new_info)
        return info

    @staticmethod
    def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Numerically stable inverse hyperbolic tangent.

        Args:
          x: Input tensor in ``[-1, 1]``.
          eps: Safety factor preventing saturation at exactly ``+-1``.

        Returns:
          Transformed tensor in unconstrained space.
        """
        x = (1 - eps) * x
        return 0.5 * torch.log((1 + x) / (1 - x))

    def x_w(self, w: torch.Tensor) -> torch.Tensor:
        """Map unconstrained variable ``w`` to valid input space.

        Args:
          w: Unconstrained optimization variable.

        Returns:
          Input-space tensor.
        """
        return self.box_scale * torch.tanh(w) + self.box_bias

    def w_x(self, x: torch.Tensor) -> torch.Tensor:
        """Map input-space tensor to unconstrained ``w`` space.

        Args:
          x: Input-space tensor.

        Returns:
          Unconstrained optimization variable.
        """
        return self.atanh((x - self.box_bias) / self.box_scale)

    def f(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Carlini-Wagner classification objective term.

        Args:
          z: Model logits.
          target: Target labels.

        Returns:
          Per-sample objective value where ``0`` implies successful attack.
        """
        if self.is_binary:
            z_t = z.clone()
            z_t[target == 0] *= -1
            z_other = 0
        else:
            idx = torch.arange(0, z.shape[0], device=z.device)
            z_t = z[idx, target]
            z_clone = z.clone()
            z_clone[idx, target] = -1e10
            z_other = torch.max(z_clone, dim=-1)[0]

        if self.targeted:
            f = F.relu(z_other - z_t + self.confidence)  # max(0, z_other-z_target+k)
        else:
            f = F.relu(z_t - z_other + self.confidence)  # max(0, z_target-z_other+k)
        return f

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples.

        Args:
          input: Clean input batch.
          target: Labels or attack targets.

        Returns:
          Adversarial batch.
        """
        raise NotImplementedError()

"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .adv_attack import AdvAttack


class RandFGSMAttack(AdvAttack):
    """Randomized FGSM attack.

    Attributes:
      eps: Total L-infinity perturbation budget.
      alpha: Random initialization magnitude.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        alpha: float,
        loss: Optional[nn.Module] = None,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> None:
        """Initialize randomized FGSM attack.

        Args:
          model: Model under attack.
          eps: Total L-infinity perturbation budget.
          alpha: Random initialization magnitude.
          loss: Loss module used to compute gradients.
          targeted: Whether the attack is targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.

        Returns:
          None.
        """
        super().__init__(model, loss, targeted, range_min, range_max)
        if eps <= 0:
            raise ValueError(f"rand-fgsm requires eps > 0, got eps={eps}")
        if alpha <= 0:
            raise ValueError(f"rand-fgsm requires alpha > 0, got alpha={alpha}")
        if alpha >= eps:
            raise ValueError(
                f"rand-fgsm requires alpha < eps, got alpha={alpha}, eps={eps}"
            )
        self.eps = eps
        self.alpha = alpha

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return attack metadata.

        Args:
          None.

        Returns:
          Dictionary describing random-FGSM configuration.
        """
        info = super().attack_info
        new_info = {
            "eps": self.eps,
            "alpha": self.alpha,
            "threat_model": "linf",
            "attack_type": "rand-fgsm",
        }
        info.update(new_info)
        return info

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate random-FGSM adversarial examples.

        Args:
          input: Clean input batch.
          target: Labels or attack targets.

        Returns:
          Adversarial batch.
        """

        x = input + self.alpha * torch.randn_like(input).sign()
        x.requires_grad = True

        output = self.model(x)
        loss = self.loss(output, target)
        self.model.zero_grad()
        loss.backward()
        dL_x = x.grad.data

        f = 1
        if self.targeted:
            f = -1

        adv_ex = x + f * (self.eps - self.alpha) * dL_x.sign()
        return self._clamp(adv_ex)

"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .adv_attack import AdvAttack


class FGSMAttack(AdvAttack):
    """Fast Gradient Sign Method attack.

    Attributes:
      eps: L-infinity perturbation budget.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float,
        loss: Optional[nn.Module] = None,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> None:
        """Initialize FGSM attack.

        Args:
          model: Model under attack.
          eps: L-infinity perturbation budget.
          loss: Loss module used to compute gradients.
          targeted: Whether the attack is targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.

        Returns:
          None.
        """
        super().__init__(model, loss, targeted, range_min, range_max)
        if eps <= 0:
            raise ValueError(f"fgsm requires eps > 0, got eps={eps}")
        self.eps = eps

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return attack metadata.

        Args:
          None.

        Returns:
          Dictionary describing FGSM configuration.
        """
        info = super().attack_info
        new_info = {"eps": self.eps, "threat_model": "linf", "attack_type": "fgsm"}
        info.update(new_info)
        return info

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate FGSM adversarial examples.

        Args:
          input: Clean input batch.
          target: Labels or attack targets.

        Returns:
          Adversarial batch.
        """

        input.requires_grad = True
        output = self.model(input)
        loss = self.loss(output, target)
        self.model.zero_grad()
        loss.backward()
        dL_x = input.grad.data

        f = 1
        if self.targeted:
            f = -1

        adv_ex = input + f * self.eps * dL_x.sign()
        return self._clamp(adv_ex)

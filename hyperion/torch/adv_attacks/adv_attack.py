"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class AdvAttack:
    """Base class for adversarial attacks.

    Attributes:
      model: Model under attack.
      loss: Loss function used to optimize perturbations.
      targeted: Whether the attack is targeted.
      range_min: Optional lower clamp bound for adversarial examples.
      range_max: Optional upper clamp bound for adversarial examples.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: Optional[nn.Module] = None,
        targeted: bool = True,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> None:
        """Initialize attack configuration.

        Args:
          model: Model under attack.
          loss: Loss module used by the attack.
          targeted: If ``True``, optimize toward the provided target labels.
          range_min: Optional minimum allowed value in generated examples.
          range_max: Optional maximum allowed value in generated examples.

        Returns:
          None.
        """
        self.model = model
        if loss is None:
            loss = nn.CrossEntropyLoss()
        self.loss = loss
        self.range_min = range_min
        self.range_max = range_max
        self.targeted = targeted

    def to(self, device: torch.device) -> None:
        """Move attack model parameters to a device.

        Args:
          device: Destination device.

        Returns:
          None.
        """
        self.model.to(device)

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return metadata describing the configured attack.

        Args:
          None.

        Returns:
          A dictionary with attack metadata fields.
        """
        return {"targeted": self.targeted}

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples.

        Args:
          input: Clean input batch.
          target: Target labels for attack optimization.

        Returns:
          Adversarial examples with the same shape as ``input``.
        """
        raise NotImplementedError()

    def _clamp(self, adv_ex: torch.Tensor) -> torch.Tensor:
        """Clamp adversarial examples to configured value range.

        Args:
          adv_ex: Candidate adversarial examples.

        Returns:
          Clamped adversarial examples.
        """
        if self.range_min is not None or self.range_max is not None:
            adv_ex = torch.clamp(adv_ex, min=self.range_min, max=self.range_max)

        return adv_ex

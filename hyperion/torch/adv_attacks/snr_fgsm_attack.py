"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .adv_attack import AdvAttack


class SNRFGSMAttack(AdvAttack):
    """FGSM-like attack that targets a requested SNR.

    Attributes:
      snr: Desired signal-to-noise ratio in dB.
    """

    def __init__(
        self,
        model: nn.Module,
        snr: float,
        loss: Optional[nn.Module] = None,
        targeted: bool = False,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ) -> None:
        """Initialize SNR-constrained FGSM attack.

        Args:
          model: Model under attack.
          snr: Desired SNR in dB for generated perturbations.
          loss: Loss module used to compute gradients.
          targeted: Whether the attack is targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.

        Returns:
          None.
        """
        super().__init__(model, loss, targeted, range_min, range_max)
        self.snr = snr

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return attack metadata.

        Args:
          None.

        Returns:
          Dictionary describing SNR-FGSM configuration.
        """
        info = super().attack_info
        new_info = {"snr": self.snr, "threat_model": "snr", "attack_type": "snr-fgsm"}
        info.update(new_info)
        return info

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate SNR-FGSM adversarial examples.

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

        dim = tuple(i for i in range(1, input.dim()))
        power_floor = 1e-12
        signal_power = torch.mean(input**2, dim=dim, keepdim=True)
        P_x = 10 * torch.log10(torch.clamp(signal_power, min=power_floor))

        noise = dL_x.sign()
        noise_power = torch.mean(noise**2, dim=dim, keepdim=True)
        P_n = 10 * torch.log10(torch.clamp(noise_power, min=power_floor))

        snr_0 = P_x - P_n
        dsnr = self.snr - snr_0
        eps = 10 ** (-dsnr / 20)

        f = 1
        if self.targeted:
            f = -1

        adv_ex = input + f * eps * noise
        return self._clamp(adv_ex)

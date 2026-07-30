"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .carlini_wagner import CarliniWagner


class CarliniWagnerL2(CarliniWagner):
    """Carlini-Wagner L2 (or SNR) attack.

    Attributes:
      binary_search_steps: Number of binary-search steps for ``c``.
      repeat: Whether to repeat last search step with upper bound.
    """

    def __init__(
        self,
        model: nn.Module,
        confidence: float = 0.0,
        lr: float = 1e-2,
        binary_search_steps: int = 9,
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
        """Initialize CW-L2 attack.

        Args:
          model: Model under attack.
          confidence: Confidence margin in the objective.
          lr: Optimizer learning rate.
          binary_search_steps: Number of binary-search steps for ``c``.
          max_iter: Maximum optimization steps per search step.
          abort_early: Whether to stop each inner optimization early.
          initial_c: Initial weight for classification objective.
          norm_time: Whether to normalize norms by time length.
          time_dim: Time-axis index used by ``norm_time``.
          use_snr: Whether to optimize SNR-based objective.
          targeted: Whether the attack is targeted.
          range_min: Optional minimum clamp value.
          range_max: Optional maximum clamp value.

        Returns:
          None.
        """

        super().__init__(
            model,
            confidence=confidence,
            lr=lr,
            max_iter=max_iter,
            abort_early=abort_early,
            initial_c=initial_c,
            norm_time=norm_time,
            time_dim=time_dim,
            use_snr=use_snr,
            targeted=targeted,
            range_min=range_min,
            range_max=range_max,
        )

        self.binary_search_steps = binary_search_steps
        self.repeat = binary_search_steps >= 10

    @property
    def attack_info(self) -> Dict[str, Any]:
        """Return attack metadata.

        Args:
          None.

        Returns:
          Dictionary describing CW-L2 configuration.
        """
        info = super().attack_info
        if self.use_snr:
            threat = "snr"
        else:
            threat = "l2"
        new_info = {
            "binary_search_steps": self.binary_search_steps,
            "threat_model": threat,
            "attack_type": "cw-l2",
        }
        info.update(new_info)
        return info

    @staticmethod
    def _compute_negsnr(x_norm: torch.Tensor, d_norm: torch.Tensor) -> torch.Tensor:
        """Compute negative SNR objective in dB.

        Args:
          x_norm: Input norm term.
          d_norm: Perturbation norm term.

        Returns:
          Negative SNR tensor.
        """
        floor = 1e-12
        d_norm = torch.clamp(d_norm, min=floor)
        x_norm = torch.clamp(x_norm, min=floor)
        return 20 * (torch.log10(d_norm) - torch.log10(x_norm))

    def generate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate CW-L2 adversarial examples.

        Args:
          input: Clean input batch.
          target: Labels or attack targets.

        Returns:
          Adversarial batch.
        """

        if self.is_binary is None:
            # run the model to know weather is binary classification problem or multiclass
            z = self.model(input)
            if z.shape[-1] == 1:
                self.is_binary = True
            else:
                self.is_binary = False
            del z

        norm_dim = tuple([i for i in range(1, input.dim())])

        if self.use_snr:
            x_norm = torch.norm(input, dim=norm_dim)
        elif self.norm_time:
            if not (-input.dim() <= self.time_dim < input.dim()):
                raise ValueError(
                    f"cw-l2 time_dim={self.time_dim} out of range for input.dim={input.dim()}"
                )

        w0 = self.w_x(input).detach()  # transform x into tanh space

        batch_size = input.shape[0]
        global_best_norm = 1e10 * torch.ones(batch_size, device=input.device)
        global_success = torch.zeros(batch_size, dtype=torch.bool, device=input.device)
        best_adv = input.clone()

        c_lower_bound = torch.zeros(batch_size, device=w0.device)
        c_upper_bound = 1e10 * torch.ones(batch_size, device=w0.device)
        c = self.initial_c * torch.ones(batch_size, device=w0.device)

        for bs_step in range(self.binary_search_steps):

            if self.repeat and bs_step == self.binary_search_steps - 1:
                # The last iteration (if we run many steps) repeat the search once.
                c = c_upper_bound

            logging.info(
                "---carlini-wagner bin-search-step={}, c={}".format(bs_step, c)
            )

            modifier = 1e-3 * torch.randn_like(w0).detach()
            modifier.requires_grad = True
            opt = optim.Adam([modifier], lr=self.lr)
            loss_prev = 1e10
            best_norm = 1e10 * torch.ones(batch_size, device=w0.device)
            success = torch.zeros(batch_size, dtype=torch.bool, device=w0.device)
            log_interval = max(1, self.max_iter // 10)
            for opt_step in range(self.max_iter):

                opt.zero_grad()
                w = w0 + modifier
                x_adv = self.x_w(w)
                z = self.model(x_adv)
                f = self.f(z, target)
                delta = x_adv - input
                d_norm = torch.norm(delta, dim=norm_dim)
                if self.use_snr:
                    # minimize the negative SNR(dB)
                    d_norm = self._compute_negsnr(x_norm, d_norm)
                elif self.norm_time:
                    # normalize by number of samples to get rms value
                    logging.info(
                        "rms {} {}".format(
                            input.shape, math.sqrt(float(input.shape[self.time_dim]))
                        )
                    )
                    d_norm = d_norm / math.sqrt(float(input.shape[self.time_dim]))

                loss1 = d_norm.mean()
                loss2 = (c * f).mean()
                loss = loss1 + loss2

                loss.backward()
                opt.step()

                # if the attack is successful f(x+delta)==0
                step_success = f < 0.0001

                # find elements that reduced l2 and where successful for current c value
                improv_idx = (d_norm < best_norm) & step_success
                best_norm[improv_idx] = d_norm[improv_idx]
                success[improv_idx] = 1

                # find elements that reduced l2 and where successful for global optimization
                improv_idx = (d_norm < global_best_norm) & step_success
                global_best_norm[improv_idx] = d_norm[improv_idx]
                global_success[improv_idx] = 1
                best_adv[improv_idx] = x_adv[improv_idx]

                if opt_step % log_interval == 0:
                    logging.info(
                        "----carlini-wagner bin-search-step={0:d}, "
                        "opt-step={1:d}/{2:d} "
                        "loss={3:.2f} d_norm={4:.2f} cf={5:.4f} "
                        "num-success={6:d}".format(
                            bs_step,
                            opt_step,
                            self.max_iter,
                            loss.item(),
                            loss1.item(),
                            loss2.item(),
                            torch.sum(step_success),
                        )
                    )

                    logging.info(
                        "----carlini-wagner bin-search-step={}, "
                        "opt-step={}/{} "
                        "step_success={}, success={} best_norm={} "
                        "global_success={} "
                        "global_best_norm={} d_norm={}".format(
                            bs_step,
                            opt_step,
                            self.max_iter,
                            step_success,
                            success,
                            best_norm,
                            global_success,
                            global_best_norm,
                            d_norm,
                        )
                    )

                loss_it = loss.item()
                if self.abort_early:
                    if loss_it > 0.999 * loss_prev:
                        logging.info(
                            "----carlini-wagner abort-early "
                            "bin-search-step={}, opt-step={}/{} "
                            "loss={}, loss_prev={}".format(
                                bs_step, opt_step, self.max_iter, loss_it, loss_prev
                            )
                        )
                        break
                    loss_prev = loss_it

            # readjust c
            c_upper_bound[success] = torch.min(c_upper_bound[success], c[success])
            c_lower_bound[~success] = torch.max(c_lower_bound[~success], c[~success])
            avg_c_idx = c_upper_bound < 1e9
            c[avg_c_idx] = (c_lower_bound[avg_c_idx] + c_upper_bound[avg_c_idx]) / 2
            cx10_idx = (~success) & (~avg_c_idx)
            c[cx10_idx] *= 10

        return best_adv

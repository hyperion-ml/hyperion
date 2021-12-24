"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .carlini_wagner import CarliniWagner


class CarliniWagnerLInf(CarliniWagner):
    def __init__(
        self,
        model,
        confidence=0.0,
        lr=1e-2,
        max_iter=10000,
        abort_early=True,
        initial_c=1e-3,
        reduce_c=False,
        c_incr_factor=2,
        tau_decr_factor=0.9,
        targeted=False,
        range_min=None,
        range_max=None,
    ):

        super().__init__(
            model,
            confidence=confidence,
            lr=lr,
            max_iter=max_iter,
            abort_early=abort_early,
            initial_c=initial_c,
            targeted=targeted,
            range_min=range_min,
            range_max=range_max,
        )
        self.reduce_c = reduce_c
        self.c_incr_factor = c_incr_factor
        self.tau_decr_factor = tau_decr_factor

    @property
    def attack_info(self):
        info = super().attack_info
        new_info = {
            "reduce_c": self.reduce_c,
            "c_incr_factor": self.c_incr_factor,
            "tau_decr_factor": self.tau_decr_factor,
            "threat_model": "linf",
            "attack_type": "cw-linf",
        }
        info.update(new_info)
        return info

    def _attack(self, x, target, start_adv, tau, c):

        w_start = self.w_x(start_adv).detach()
        c_step = 0
        modifier = 1e-3 * torch.randn_like(w_start).detach()
        modifier.requires_grad = True
        opt = optim.Adam([modifier], lr=self.lr)
        while c < 2e6:
            step_success = False
            for opt_step in range(self.max_iter):

                opt.zero_grad()
                w = w_start + modifier
                x_adv = self.x_w(w)
                z = self.model(x_adv)
                f = self.f(z, target)
                delta = x_adv - x
                r = torch.abs(delta) - tau
                loss1 = F.relu(r).mean()
                loss2 = (c * f).mean()
                loss = loss1 + loss2
                loss.backward()
                opt.step()

                # if the attack is successful f(x+delta)==0
                step_success = f < 1e-4
                if opt_step % (self.max_iter // 10) == 0:
                    logging.info(
                        "--------carlini-wagner-linf--l1-optim "
                        "c_step={0:d} opt-step={1:d} c={2:f} "
                        "loss={3:.2f} d_norm={4:.2f} cf={5:.5f} "
                        "success={6}".format(
                            c_step,
                            opt_step,
                            c,
                            loss.item(),
                            loss1.item() + tau,
                            loss2.item(),
                            bool(step_success.item()),
                        )
                    )

                loss_it = loss.item()
                if loss_it <= 0 or (step_success and self.abort_early):
                    break

            if step_success:
                return x_adv.detach(), c

            c *= self.c_incr_factor
            c_step += 1

        return None

    def _generate_one(self, x, target):

        x = x.unsqueeze(dim=0)
        target = target.unsqueeze(dim=0)

        best_adv = x
        c = self.initial_c
        tau_max = max(abs(self.range_max), abs(self.range_min))
        tau_min = 1.0 / 256
        tau = tau_max
        cur_it = 0
        while tau > tau_min:
            res = self._attack(x, target, best_adv, tau, c)
            if res is None:
                logging.info(
                    "----carlini-wagner-linf--return it={} x-shape={} "
                    "tau={} c={}".format(cur_it, x.shape, tau, c)
                )
                return best_adv[0]

            x_adv, c = res
            if self.reduce_c:
                c /= 2

            actual_tau = torch.max(torch.abs(x - x_adv))
            if actual_tau < tau:
                tau = actual_tau

            logging.info(
                "----carlini-wagner-lin--tau-optim it={} x-shape={} "
                "tau={}".format(cur_it, x.shape, tau)
            )

            best_adv = x_adv
            tau *= self.tau_decr_factor
            cur_it += 1

        return best_adv[0]

    def generate(self, input, target):

        if self.is_binary is None:
            # run the model to know weather is binary classification problem or multiclass
            z = self.model(input)
            if z.shape[-1] == 1:
                self.is_binary = True
            else:
                self.is_binary = None
            del z

        x_adv = input.clone()
        for i in range(input.shape[0]):
            x_adv[i] = self._generate_one(input[i], target[i])

        return x_adv

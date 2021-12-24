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


class CarliniWagnerL0(CarliniWagner):
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
        indep_channels=False,
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
        self.indep_channels = indep_channels

    @property
    def attack_info(self):
        info = super().attack_info
        new_info = {
            "reduce_c": self.reduce_c,
            "c_incr_factor": self.c_incr_factor,
            "indep_channel": self.indep_channels,
            "threat_model": "l0",
            "attack_type": "cw-l0",
        }
        info.update(new_info)
        return info

    def _attack_l2(self, x, target, valid, start_adv, c):

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
                x_adv = self.x_w(w) * valid + (1 - valid) * x
                z = self.model(x_adv)
                f = self.f(z, target)
                delta = x_adv - x
                d_norm = torch.norm(delta)
                loss1 = d_norm.mean()
                loss2 = (c * f).mean()
                loss = loss1 + loss2

                loss.backward()
                opt.step()

                # if the attack is successful f(x+delta)==0
                step_success = f < 1e-4
                if opt_step % (self.max_iter // 10) == 0:
                    logging.info(
                        "--------carlini-wagner-l0--l2-optim c_step={0:d} opt-step={1:d} c={2:f} "
                        "loss={3:.2f} d_norm={4:.2f} cf={5:.5f} success={6}".format(
                            c_step,
                            opt_step,
                            c,
                            loss.item(),
                            loss1.item(),
                            loss2.item(),
                            bool(step_success.item()),
                        )
                    )

                loss_it = loss.item()
                if step_success and self.abort_early:
                    break

            if step_success:
                grad = modifier.grad.data * valid
                return x_adv.detach(), grad, c

            c *= self.c_incr_factor
            c_step += 1

        return None

    def _generate_one(self, x, target):

        x = x.unsqueeze(dim=0)
        target = target.unsqueeze(dim=0)

        valid = torch.ones_like(x)
        best_adv = x
        c = self.initial_c
        cur_it = 0
        l0 = 0
        while True:
            res = self._attack_l2(x, target, valid, best_adv, c)
            if res is None:
                logging.info(
                    "----carlini-wagner-l0--return it={} x-shape={} l0={} c={}".format(
                        cur_it, x.shape, l0, c
                    )
                )
                return best_adv[0]

            x_adv, dy_x, c = res
            if torch.sum(valid) == 0:
                # if no pixels changed, return
                return x

            if self.reduce_c:
                c /= 2

            d_x = torch.abs(x - x_adv)
            if x.dim() == 2:
                # audio
                l0 = x.shape[1] - torch.sum(d_x < 0.0001)
                total_change = torch.abs(dy_x) * d_x
                valid = torch.flatten(valid)
                valid_all = valid
            elif x.dim() == 4:
                # image
                l0 = x.shape[2] * x.shape[3] - torch.sum(
                    torch.sum(d_x < 0.0001, dim=2) > 0
                )

                if self.indep_channels:
                    total_change = torch.abs(dy_x) * d_x
                    valid = torch.flatten(valid)
                    valid_all = valid
                else:
                    total_change = torch.sum(torch.abs(dy_x, dim=1)) * torch.sum(
                        d_x, dim=1
                    )
                    valid = valid.view(x.shape[1], x.shape[2] * x.shape[3])
                    valid_all = torch.sum(valid, dim=0) > 0
            else:
                raise NotImplementedError()

            total_change = torch.flatten(total_change)
            change_count = 0
            cur_num_valid = torch.sum(valid_all)
            avg_change = torch.mean(total_change)
            max_change = torch.max(total_change)
            l0 = float(l0)
            for idx in torch.argsort(total_change):
                if valid_all[idx]:
                    change_count += 1
                    if valid.dim() == 1:
                        valid[idx] = 0
                    else:
                        valid[:, idx] = 0

                    # if total_change[idx] > .01 #this is what is hard coded in carlini's code but this makes optim very slow, it just removes one sample at a time, not feasible for speech
                    if total_change[idx] > 0.5 * max_change:
                        # if change is big we stop putting elements to 0
                        logging.info(
                            "break because of large total-change "
                            "{} > {}".format(total_change[idx], 0.5 * max_change)
                        )
                        break

                    if change_count >= 0.5 * l0:  # in carlini's code 0.3*l0**.5
                        # if we put to many elements to 0, we stop
                        logging.info(
                            "break because large change-count "
                            "{} >= {} l0={}".format(change_count, 0.5 * float(l0), l0)
                        )
                        break

            logging.info(
                "----carlini-wagner-l0--l0-optim it={} x-shape={} "
                "l0={} c={}"
                "cur-num-valid-changes={} next-num-valid-changes={} "
                "avg-total-change={} "
                "max-total-change={} ".format(
                    cur_it,
                    x.shape,
                    l0,
                    c,
                    cur_num_valid,
                    cur_num_valid - change_count,
                    avg_change,
                    max_change,
                )
            )

            valid = valid.view_as(x)
            best_adv = x_adv
            cur_it += 1

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

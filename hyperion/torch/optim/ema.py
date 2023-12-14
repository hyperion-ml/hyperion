"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import math

import torch
from jsonargparse import ActionParser, ArgumentParser


class ExpMovingAvg:
    def __init__(
        self, params, init_momentum=0.996, momentum=0.996, warmup_steps=0, global_step=0
    ):
        if not isinstance(params, list):
            params = [params]
        self.params = [list(p) for p in params]
        self.init_momentum = init_momentum
        self._momentum = momentum
        self.warmup_steps = warmup_steps
        self.global_step = global_step

    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict` needed to restart the training."""
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    @property
    def momentum(self):
        if self.global_step >= self.warmup_steps:
            return self._momentum
        else:
            alpha = (1 + math.cos(self.global_step / self.warmup_steps * math.pi)) / 2
            return self.init_momentum * alpha + self._momentum * (1 - alpha)

    @torch.no_grad()
    def step(self, new_params):
        if not isinstance(new_params, list):
            new_params = [new_params]

        assert len(self.params) == len(new_params)
        momentum = self.momentum
        for param_group, new_param_group in zip(self.params, new_params):
            for p, p_new in zip(param_group, new_param_group):
                p.data.mul_(momentum).add_((1 - momentum) * p_new.data)

        self.global_step += 1

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--init-momentum", default=0.996, type=float, help="initial momentum"
        )
        parser.add_argument(
            "--momentum", default=0.996, type=float, help="final momentum"
        )
        parser.add_argument(
            "--warmup-steps", default=0, type=int, help="momentum warmup steps"
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

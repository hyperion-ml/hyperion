"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


import torch
from torch.optim.optimizer import Optimizer


class FGSM(Optimizer):
    """Implements Fast Gradient Sign Method"""

    def __init__(self, params, epsilon):
        defaults = dict(epsilon=epsilon)
        super(FGSM, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(-epsilon, d_p.sign())

        return loss

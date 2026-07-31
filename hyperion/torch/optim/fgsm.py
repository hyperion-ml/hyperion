"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

ParamsT = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class FGSM(Optimizer):
    """Fast Gradient Sign Method (FGSM) optimizer.

    This optimizer applies a fixed-size sign step to each parameter:
    ``p <- p - epsilon * sign(grad)``.
    """

    def __init__(self, params: ParamsT, epsilon: float) -> None:
        """Initialize FGSM optimizer.

        Args:
            params: Iterable of parameters or optimizer parameter-group dicts.
            epsilon: Fixed step magnitude used for sign updates.

        Raises:
            ValueError: If ``epsilon`` is negative.
        """
        if epsilon < 0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        defaults = dict(epsilon=epsilon)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        """Performs a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the
                loss tensor.

        Returns:
            The loss value returned by ``closure`` when provided, else ``None``.

        Raises:
            RuntimeError: If a sparse gradient is encountered.
        """
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FGSM does not support sparse gradients")
                p.add_(grad.sign(), alpha=-epsilon)

        return loss

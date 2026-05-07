"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

#

from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch.nn as nn

from .snake import Snake1d
from .swish import DoubleSwish, DoubleSwish6, Swish, Swish6

ActivationConfig = Dict[str, Any]
ActivationCtor = Callable[..., nn.Module]
ActivationSpec = Optional[Union[str, Mapping[str, Any], Callable[..., nn.Module]]]


act_dict: Dict[str, ActivationCtor] = {
    "elu": nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hardtanh": nn.Hardtanh,
    "leakyrelu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "prelu": nn.PReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": nn.Threshold,
    "softmin": nn.Softmin,
    "softmax": nn.Softmax,
    "softmax2d": nn.Softmax2d,
    "logsoftmax": nn.LogSoftmax,
    "alogsoftmax": nn.AdaptiveLogSoftmaxWithLoss,
    "swish": Swish,
    "double_swish": DoubleSwish,
    "swish6": Swish6,
    "double_swish6": DoubleSwish6,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "snake1d": Snake1d,
}


class ActivationFactory:
    """Factory utilities for constructing and serializing activation modules.

    Examples:
        >>> act = ActivationFactory.create("relu")
        >>> isinstance(act, nn.ReLU)
        True

        >>> act = ActivationFactory.create({"name": "leakyrelu", "negative_slope": 0.2})
        >>> float(act.negative_slope)
        0.2

        >>> cfg = ActivationFactory.get_config(nn.SiLU(inplace=False))
        >>> cfg["name"], cfg["inplace"]
        ('silu', False)
    """

    @staticmethod
    def create(activation: ActivationSpec, **kwargs: Any) -> Optional[nn.Module]:
        """Creates an activation module from a flexible specification.

        Args:
            activation: Activation specification. Supported values are:
                - ``None``: returns ``None``.
                - ``str``: activation name key in :data:`act_dict`.
                - mapping: configuration with ``"name"`` plus constructor kwargs.
                - callable: constructor returning an ``nn.Module``.
            **kwargs: Extra constructor kwargs used when ``activation`` is a string.

        Returns:
            Instantiated activation module, or ``None`` when ``activation`` is
            ``None``.
        """

        if activation is None:
            return None

        if isinstance(activation, str):
            return ActivationFactory.create_from_str(activation, **kwargs)

        if isinstance(activation, Mapping):
            name = activation["name"]
            kwargs = dict(activation)
            del kwargs["name"]
            return ActivationFactory.create_from_str(name, **kwargs)

        return activation(**kwargs)

    @staticmethod
    def create_from_str(activation_name: str, **kwargs: Any) -> nn.Module:
        """Creates an activation module from its registered name.

        Args:
            activation_name: Activation name key in :data:`act_dict`.
            **kwargs: Extra arguments forwarded to the activation constructor.

        Returns:
            Instantiated activation module.

        Notes:
            If ``inplace`` is not provided, the factory first tries to call the
            constructor with ``inplace=True`` and silently retries without it if
            unsupported. For ``leakyrelu``, ``negative_slope`` defaults to ``0.1``.
        """

        if "inplace" not in kwargs:
            # try to make it inplace anyway
            kwargs["inplace"] = True
            try:
                return act_dict[activation_name](**kwargs)
            except TypeError as e:
                # Retry without inplace only when the constructor rejects it.
                if "inplace" not in str(e):
                    raise
                del kwargs["inplace"]

        if activation_name == "leakyrelu":
            # LeakyReLU has a negative_slope argument
            if "negative_slope" not in kwargs:
                kwargs["negative_slope"] = 0.1

        return act_dict[activation_name](**kwargs)

    @staticmethod
    def get_config(activation: nn.Module) -> Optional[ActivationConfig]:
        """Returns a serializable configuration for a known activation module.

        Args:
            activation: Instantiated activation module.

        Returns:
            A configuration dictionary compatible with :meth:`create` input when
            the activation type is recognized, otherwise ``None``.
        """
        if isinstance(activation, nn.ELU):
            return {
                "name": "elu",
                "alpha": activation.alpha,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.ReLU6):
            return {"name": "relu6", "inplace": activation.inplace}
        if isinstance(activation, nn.Hardshrink):
            return {"name": "hardshrink", "lambd": activation.lambd}
        if isinstance(activation, nn.Hardtanh):
            return {
                "name": "hardtanh",
                "min_val": activation.min_val,
                "max_val": activation.max_val,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.LeakyReLU):
            return {
                "name": "leakyrelu",
                "negative_slope": activation.negative_slope,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.LogSigmoid):
            return {"name": "logsigmoid"}
        if isinstance(activation, nn.PReLU):
            return {
                "name": "prelu",
                "num_parameters": activation.num_parameters,
                "init": activation.init,
            }
        if isinstance(activation, nn.ReLU):
            return {"name": "relu", "inplace": activation.inplace}
        if isinstance(activation, nn.RReLU):
            return {
                "name": "rrelu",
                "lower": activation.lower,
                "upper": activation.upper,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.SELU):
            return {"name": "selu", "inplace": activation.inplace}
        if isinstance(activation, nn.CELU):
            return {
                "name": "celu",
                "alpha": activation.alpha,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.Sigmoid):
            return {"name": "sigmoid"}
        if isinstance(activation, nn.Softplus):
            return {
                "name": "softplus",
                "beta": activation.beta,
                "threshold": activation.threshold,
            }
        if isinstance(activation, nn.Softshrink):
            return {"name": "softshrink", "lambd": activation.lambd}
        if isinstance(activation, nn.Softsign):
            return {"name": "softsign"}
        if isinstance(activation, nn.Tanh):
            return {"name": "tanh"}
        if isinstance(activation, nn.Tanhshrink):
            return {"name": "tanhshrink"}
        if isinstance(activation, nn.Threshold):
            return {
                "name": "threshold",
                "threshold": activation.threshold,
                "value": activation.value,
                "inplace": activation.inplace,
            }
        if isinstance(activation, nn.Softmin):
            return {"name": "softmin", "dim": activation.dim}
        if isinstance(activation, nn.Softmax):
            return {"name": "softmax", "dim": activation.dim}
        if isinstance(activation, nn.Softmax2d):
            return {"name": "softmax2d"}
        if isinstance(activation, nn.LogSoftmax):
            return {"name": "logsoftmax", "dim": activation.dim}
        if isinstance(activation, nn.AdaptiveLogSoftmaxWithLoss):
            return {
                "name": "alogsoftmax",
                "in_features": activation.in_features,
                "n_classes": activation.n_classes,
                "cutoffs": activation.cutoffs[:-1],
                "div_value": activation.div_value,
                "head_bias": activation.head_bias,
            }
        if isinstance(activation, Swish):
            return {"name": "swish"}
        if isinstance(activation, DoubleSwish):
            return {"name": "double_swish"}
        if isinstance(activation, Swish6):
            return {"name": "swish6"}
        if isinstance(activation, DoubleSwish6):
            return {"name": "double_swish6"}

        if isinstance(activation, nn.GELU):
            return {"name": "gelu", "approximate": activation.approximate}

        if isinstance(activation, nn.SiLU):
            return {"name": "silu", "inplace": activation.inplace}

        if isinstance(activation, Snake1d):
            return {"name": "snake1d", "channels": activation.alpha.shape[1]}

        return None

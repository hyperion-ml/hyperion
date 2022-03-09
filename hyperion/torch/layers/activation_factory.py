"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
#

import torch.nn as nn
from .swish import Swish

act_dict = {
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
}


class ActivationFactory(object):
    @staticmethod
    def create(activation, **kwargs):
        """Creates a non-linear activation object

        Args:
           activation: String with activation type,
                       dictionary with name field indicating the activation type,
                       and extra activation arguments
                       None, then it returns None,
                       Activation constructor

           **kwargs: Extra arguments for activation constructor

        Return:
           Non-linear activation object
        """

        if activation is None:
            return None

        if isinstance(activation, str):
            return ActivationFactory.create_from_str(activation, **kwargs)

        if isinstance(activation, dict):
            name = activation["name"]
            kwargs = activation.copy()
            del kwargs["name"]
            return ActivationFactory.create_from_str(name, **kwargs)

        activation = activation()

        return activation

    @staticmethod
    def create_from_str(activation_name, **kwargs):
        """Creates a non-linear activation object from string

        Args:
           activation: str with activation type,
           **kwargs: extra arguments for activation constructor

        Return:
           Non-linear activation object
        """

        if "inplace" not in kwargs:
            # try to make it inplace anyway
            kwargs["inplace"] = True
            try:
                return act_dict[activation_name](**kwargs)
            except:
                # activation didn't have inplace option
                del kwargs["inplace"]
                pass

        return act_dict[activation_name](**kwargs)

    @staticmethod
    def get_config(activation):
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
            return {"name": "simoid"}
        if isinstance(activation, nn.Softplus):
            return {
                "name": "softplus",
                "beta": activation.beta,
                "threshold": activation.threshold,
            }
        if isinstance(activation, nn.Softshrink):
            return {"name": "softshrink"}
        if isinstance(activation, nn.Softsign):
            return {"name": "softsign", "lambd": activation.lambd}
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
                "name": "asoftmax",
                "in_features": activation.in_features,
                "n_classes": activation.n_classes,
                "cutoffs": activation.cutoffs,
                "div_value": activation.div_value,
                "head_bias": activation.head_bias,
            }
        if isinstance(activation, Swish):
            return {"name": "swish"}

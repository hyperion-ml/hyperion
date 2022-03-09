"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn


class NormLayer2dFactory(object):
    """Factory class to create normalization layers for
    tensors with 2D spatial dimension.
    """

    @staticmethod
    def create(norm_name, num_groups=None, momentum=0.1, eps=1e-5):
        """Creates a layer-norm callabe constructor

        Args:
          norm_name: str with normalization layer name,
                     in [batch-norm, group-norm, instance-norm,
                         instance-norm-affine, layer-norm ]
          num_groups: num_groups for group-norm
          momentum: default momentum
          eps: default epsilon for numerical stability

        Returns:
           Callable contructor to crate layer-norm layers
        """

        # if None we assume batch-norm
        if norm_name is None or norm_name == "batch-norm":
            return lambda x, momentum=momentum, eps=eps: nn.BatchNorm2d(
                x, momentum=momentum, eps=eps
            )

        if not isinstance(norm_name, str):
            # we assume that this is already a layernorm object
            # and return unchanged
            return norm_name

        if norm_name == "group-norm":
            num_groups = 32 if num_groups is None else num_groups
            return lambda x, momentum=momentum, eps=eps: nn.GroupNorm(
                num_groups, x, eps=eps
            )

        if norm_name == "instance-norm":
            return lambda x, momentum=momentum, eps=eps: nn.InstanceNorm2d(x, eps=eps)

        if norm_name == "instance-norm-affine":
            return lambda x, momentum=momentum, eps=eps: nn.InstanceNorm2d(
                x, eps=eps, affine=True
            )

        if norm_name == "layer-norm":
            # it is equivalent to groupnorm with 1 group
            return lambda x, momentum=momentum, eps=eps: nn.GroupNorm(1, x, eps=eps)


class NormLayer1dFactory(object):
    """Factory class to create normalization layers for
    tensors with 1D spatial (time) dimension.
    """

    @staticmethod
    def create(norm_name, num_groups=None, momentum=0.1, eps=1e-5):
        """Creates a layer-norm callabe constructor

        Args:
          norm_name: str with normalization layer name,
                     in [batch-norm, group-norm, instance-norm,
                         instance-norm-affine, layer-norm ]
          num_groups: num_groups for group-norm
          momentum: default momentum
          eps: default epsilon for numerical stability

        Returns:
           Callable contructor to crate layer-norm layers
        """

        # if None we assume batch-norm
        if norm_name is None or norm_name == "batch-norm":
            return lambda x, momentum=momentum, eps=eps: nn.BatchNorm1d(
                x, momentum=momentum, eps=eps
            )

        if not isinstance(norm_name, str):
            # we assume that this is already a layernorm object
            # and return unchanged
            return norm_name

        if norm_name == "group-norm":
            num_groups = 32 if num_groups is None else num_groups
            return lambda x, momentum=momentum, eps=eps: nn.GroupNorm(
                num_groups, x, eps=eps
            )

        if norm_name == "instance-norm":
            return lambda x, momentum=momentum, eps=eps: nn.InstanceNorm1d(x, eps=eps)

        if norm_name == "instance-norm-affine":
            return lambda x, momentum=momentum, eps=eps: nn.InstanceNorm1d(
                x, eps=eps, affine=True
            )

        if norm_name == "layer-norm":
            # it is equivalent to groupnorm with 1 group
            return lambda x, momentum=momentum, eps=eps: nn.GroupNorm(1, x, eps=eps)

"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Protocol, TypeVar

import torch.nn as nn


class NormLayerCtor(Protocol):
    """Callable protocol for normalization layer constructors."""

    def __call__(
        self,
        num_channels: int,
        momentum: float = ...,
        eps: float = ...,
    ) -> nn.Module: ...


T = TypeVar("T")


class NormLayer2dFactory:
    """Factory for normalization layers operating on 2D feature maps.

    Examples:
        >>> norm_ctor = NormLayer2dFactory.create("batch-norm")
        >>> norm = norm_ctor(64)
        >>> isinstance(norm, nn.BatchNorm2d)
        True

        >>> group_ctor = NormLayer2dFactory.create("group-norm", num_groups=8)
        >>> group_norm = group_ctor(64)
        >>> isinstance(group_norm, nn.GroupNorm)
        True
    """

    @staticmethod
    def create(
        norm_name: Optional[str | T],
        num_groups: Optional[int] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> NormLayerCtor | T:
        """Builds a normalization-layer constructor for 2D inputs.

        Args:
            norm_name: Normalization identifier or pre-built object.
                Supported strings are ``"batch-norm"``, ``"group-norm"``,
                ``"instance-norm"``, ``"instance-norm-affine"``, and
                ``"layer-norm"``. ``None`` is treated as ``"batch-norm"``.
                Non-string values are returned unchanged.
            num_groups: Number of groups for ``"group-norm"``.
                Defaults to ``32`` when not provided.
            momentum: Momentum passed to batch normalization constructors.
            eps: Epsilon used for numerical stability in normalization layers.

        Returns:
            A callable that receives ``num_channels`` plus optional
            normalization kwargs (for example ``momentum`` and ``eps``),
            and returns an ``nn.Module``. Returns ``norm_name`` unchanged
            when ``norm_name`` is a non-string object.
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

        raise ValueError(f"unknown normalization layer '{norm_name}'")


class NormLayer1dFactory:
    """Factory for normalization layers operating on 1D sequences.

    Examples:
        >>> norm_ctor = NormLayer1dFactory.create("batch-norm")
        >>> norm = norm_ctor(256)
        >>> isinstance(norm, nn.BatchNorm1d)
        True

        >>> custom = nn.LayerNorm(256)
        >>> NormLayer1dFactory.create(custom) is custom
        True
    """

    @staticmethod
    def create(
        norm_name: Optional[str | T],
        num_groups: Optional[int] = None,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> NormLayerCtor | T:
        """Builds a normalization-layer constructor for 1D inputs.

        Args:
            norm_name: Normalization identifier or pre-built object.
                Supported strings are ``"batch-norm"``, ``"group-norm"``,
                ``"instance-norm"``, ``"instance-norm-affine"``, and
                ``"layer-norm"``. ``None`` is treated as ``"batch-norm"``.
                Non-string values are returned unchanged.
            num_groups: Number of groups for ``"group-norm"``.
                Defaults to ``32`` when not provided.
            momentum: Momentum passed to batch normalization constructors.
            eps: Epsilon used for numerical stability in normalization layers.

        Returns:
            A callable that receives ``num_channels`` plus optional
            normalization kwargs (for example ``momentum`` and ``eps``),
            and returns an ``nn.Module``. Returns ``norm_name`` unchanged
            when ``norm_name`` is a non-string object.
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
            return lambda x, momentum=momentum, eps=eps: nn.LayerNorm(x, eps=eps)

        raise ValueError(f"unknown normalization layer '{norm_name}'")

"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import TypeAlias, Union

import loralib as lora
import torch
import torch.nn as nn
from loralib import mark_only_lora_as_trainable

LoRACompatibleLayer: TypeAlias = Union[
    nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d
]


def repr_lora(self: nn.Module, str_base: str) -> str:
    """Builds a module repr string including LoRA-specific fields.

    Args:
        self: LoRA module instance.
        str_base: Base repr string returned by the parent class.

    Returns:
        Repr string augmented with LoRA rank, alpha, dropout, and merge state.
    """
    if isinstance(self.lora_dropout, nn.Dropout):
        lora_dropout = self.lora_dropout.p
    else:
        lora_dropout = 0

    str_lora = f", r={self.r}, alpha={self.lora_alpha}, dropout={lora_dropout}, merge_weights={self.merge_weights})"
    return str_base[:-1] + str_lora


class LinearLoRA(lora.Linear):
    def __repr__(self) -> str:
        str_base = super().__repr__()
        return repr_lora(self, str_base)


class EmbeddingLoRA(lora.Embedding):
    def __repr__(self) -> str:
        str_base = super().__repr__()
        return repr_lora(self, str_base)


class Conv1dLoRA(lora.Conv1d):
    def __repr__(self) -> str:
        str_base = super().__repr__()
        return repr_lora(self, str_base)


class Conv2dLoRA(lora.Conv2d):
    def __repr__(self) -> str:
        str_base = super().__repr__()
        return repr_lora(self, str_base)


class Conv3dLoRA(lora.Conv3d):
    def __repr__(self) -> str:
        str_base = super().__repr__()
        return repr_lora(self, str_base)


LoRALayer: TypeAlias = Union[
    EmbeddingLoRA, LinearLoRA, Conv1dLoRA, Conv2dLoRA, Conv3dLoRA
]


class LoRAFactory:
    """Factory for converting pretrained layers into LoRA-enabled layers.

    Examples:
        >>> linear = nn.Linear(256, 128)
        >>> lora_linear = LoRAFactory.create_from_pretrained(
        ...     linear, r=8, lora_alpha=16, lora_dropout=0.1
        ... )
        >>> isinstance(lora_linear, LinearLoRA)
        True

        >>> mark_only_lora_as_trainable(lora_linear)
    """

    @staticmethod
    def _copy_pretrained_params(target: nn.Module, source: nn.Module) -> None:
        """Copies pretrained base parameters from source into target.

        Args:
            target: Destination LoRA layer.
            source: Source pretrained layer.

        Returns:
            None.
        """
        target.to(device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            target.weight.copy_(source.weight)
            if source.bias is not None and target.bias is not None:
                target.bias.copy_(source.bias)

    @staticmethod
    def create_from_pretrained(
        layer: LoRACompatibleLayer,
        r: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ) -> LoRALayer:
        """Creates a LoRA layer initialized from a pretrained base layer.

        Args:
            layer: Source layer to convert. Supported types are embedding, linear,
                and 1D/2D/3D convolutions.
            r: LoRA rank.
            lora_alpha: LoRA scaling factor.
            lora_dropout: Dropout probability applied in the LoRA path.
            merge_weights: Whether to merge LoRA weights into base weights in eval mode.

        Returns:
            A LoRA-wrapped layer with copied pretrained base weights and bias.
        """
        if isinstance(layer, nn.Embedding):
            lora_layer = EmbeddingLoRA(
                layer.num_embeddings,
                layer.embedding_dim,
                padding_idx=layer.padding_idx,
                max_norm=layer.max_norm,
                norm_type=layer.norm_type,
                scale_grad_by_freq=layer.scale_grad_by_freq,
                sparse=layer.sparse,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            LoRAFactory._copy_pretrained_params(lora_layer, layer)

        elif isinstance(layer, nn.Linear):
            bias = layer.bias is not None
            lora_layer = LinearLoRA(
                layer.in_features,
                layer.out_features,
                bias=bias,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            LoRAFactory._copy_pretrained_params(lora_layer, layer)

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if isinstance(layer, nn.Conv1d):
                lora_class = Conv1dLoRA
            elif isinstance(layer, nn.Conv2d):
                lora_class = Conv2dLoRA
            elif isinstance(layer, nn.Conv3d):
                lora_class = Conv3dLoRA

            bias = layer.bias is not None
            lora_layer = lora_class(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=bias,
                padding_mode=layer.padding_mode,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            LoRAFactory._copy_pretrained_params(lora_layer, layer)

        else:
            raise TypeError(
                f"Unsupported layer type for LoRA conversion: {type(layer).__name__}"
            )

        return lora_layer

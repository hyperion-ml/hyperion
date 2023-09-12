"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Union

import loralib as lora
import torch.nn as nn
from loralib import *


class LoRAFactory:
    def create_from_pretrained(
        layer: Union[nn.Embedding, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d],
        r: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        if isinstance(layer, nn.Embedding):
            lora_layer = lora.Embedding(
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
            lora_layer.weight.data = layer.weight.data

        elif isinstance(layer, nn.Linear):
            bias = layer.bias is not None
            lora_layer = lora.Linear(
                layer.in_features,
                layer.out_features,
                bias=bias,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            lora_layer.weight.data = layer.weight.data
            if bias:
                lora_layer.bias.data = layer.bias.data

        elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if isinstance(layer, nn.Conv1d):
                lora_class = lora.Conv1d
            elif isinstance(layer, nn.Conv2d):
                lora_class = lora.Conv2d
            elif isinstance(layer, nn.Conv3d):
                lora_class = lora.Conv3d

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
            lora_layer.weight.data = layer.weight.data
            if bias:
                lora_layer.bias.data = layer.bias.data

        return lora_layer

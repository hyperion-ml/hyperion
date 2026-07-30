"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from ...utils.misc import filter_func_args
from ..layer_blocks.hifi_blocks import HiFiBlock
from ..layers import ActivationFactory as AF
from ..utils import scale_seq_lengths, seq_lengths_to_mask
from .net_arch import NetArch


class HiFiGenerator(NetArch):
    """
    HiFi-GAN Generator architecture based on WaveNet-style dilated convolutions.

    This generator upsamples a low-resolution sequence (e.g., mel-spectrogram)
    into a high-fidelity audio waveform using transposed convolutions and
    residual blocks (HiFiBlocks).

    Args:
        in_feats (int): Number of input channels (e.g., mel bins).
        out_feats (int): Number of output channels (e.g., 1 for mono audio).
        resb_kernel_sizes (List[int]): List of kernel sizes for HiFiBlocks.
        resb_dilations (List[int]): List of dilation configurations per HiFiBlock.
        upsample_init_channels (int): Number of channels before the first upsample layer.
        upsample_kernel_sizes (List[int]): Kernel sizes for transposed conv layers.
        upsample_strides (List[int]): Stride values for each upsampling stage.
        activation (Union[str, Dict[str, Any], nn.Module], optional): Activation function to use. Default is 'leakyrelu'.
        cond_channels (int, optional): Number of conditioning channels. Default is 0.

    Attributes:
        in_feats (int): Number of input feature channels.
        out_feats (int): Number of output channels.
        resb_kernel_sizes (List[int]): Kernel sizes used in residual blocks.
        resb_dilations (List[int]): Dilation pattern used in residual blocks.
        upsample_init_channels (int): Number of channels before the first upsample layer.
        upsample_kernel_sizes (List[int]): Kernel sizes for transposed-convolution upsampling layers.
        upsample_strides (List[int]): Stride values for each upsampling stage.
        cond_channels (int): Number of conditioning channels.
        activation (nn.Module): Activation module applied between convolutions.
        num_kernels (int): Number of residual blocks per upsampling stage.
        num_upsamples (int): Number of upsampling stages.
        in_conv (nn.Conv1d): Initial projection convolution.
        upsample_layers (nn.ModuleList): Transposed-convolution upsampling layers.
        blocks (nn.ModuleList): Residual HiFi blocks.
        out_conv (nn.Conv1d): Final output projection convolution.
        cond_layer (Optional[nn.Conv1d]): Optional conditioning projection layer.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int = 1,
        resb_kernel_sizes: List[int] = [3, 7, 11],
        resb_dilations: List[int] = [1, 3, 5],
        upsample_init_channels: int = 512,
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_strides: List[int] = [10, 8, 2, 2],
        activation: Union[str, Dict[str, Any], nn.Module] = "leakyrelu",
        cond_channels: int = 0,
    ) -> None:
        """
        Initialize the HiFi generator.

        Args:
            in_feats: Number of input feature channels.
            out_feats: Number of output channels.
            resb_kernel_sizes: Kernel sizes for the residual blocks.
            resb_dilations: Dilation pattern used in each residual block.
            upsample_init_channels: Number of channels before the first upsampling stage.
            upsample_kernel_sizes: Kernel sizes for each transposed-convolution layer.
            upsample_strides: Stride values for each upsampling layer.
            activation: Activation specification passed to
                :func:`~hyperion.torch.layers.activation_factory.ActivationFactory.create`.
            cond_channels: Number of conditioning channels. Set to ``0`` to disable conditioning.
        """
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.resb_kernel_sizes = resb_kernel_sizes
        self.resb_dilations = resb_dilations
        self.upsample_init_channels = upsample_init_channels
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_strides = upsample_strides
        self.cond_channels = cond_channels

        if activation == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            activation = AF.create(activation)

        self.activation = activation

        self.num_kernels = len(resb_kernel_sizes)
        self.num_upsamples = len(upsample_strides)
        assert len(upsample_kernel_sizes) == len(upsample_strides)

        self.in_conv = nn.Conv1d(
            in_channels=in_feats,
            out_channels=upsample_init_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        self.upsample_layers = nn.ModuleList()
        for i, (stride, kernel) in enumerate(
            zip(upsample_strides, upsample_kernel_sizes)
        ):
            in_ch = upsample_init_channels // (2**i)
            out_ch = upsample_init_channels // (2 ** (i + 1))
            conv = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=(kernel - stride) // 2,
            )
            self.upsample_layers.append(weight_norm(conv))

        self.blocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_init_channels // (2 ** (i + 1))
            for k in resb_kernel_sizes:
                self.blocks.append(
                    HiFiBlock(
                        channels,
                        kernel_size=k,
                        dilations=resb_dilations,
                        activation=activation,
                    )
                )

        self.out_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=out_feats,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )

        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(
                cond_channels, upsample_init_channels, kernel_size=1
            )
        else:
            self.cond_layer = None

        self.init_weights()

    @property
    def stride(self) -> int:
        """
        Returns the stride of the generator.

        Returns:
            int: Product of all upsample strides.
        """
        return math.prod(self.upsample_strides)

    def init_weights(self) -> None:
        """
        Initialize all convolutional weights with normal distribution.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                if (
                    is_parametrized(m)
                    and hasattr(m, "parametrizations")
                    and "weight" in m.parametrizations
                ):
                    g = m.parametrizations.weight.original0
                    v = m.parametrizations.weight.original1
                    nn.init.normal_(v, 0.0, 0.01)
                    with torch.no_grad():
                        g.copy_(v.flatten(1).norm(dim=1, keepdim=True).view_as(g))
                else:
                    nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the HiFi-GAN generator.

        Args:
            x (Tensor): Input tensor of shape (B, T, in_feats).
            x_lengths (Tensor, optional): Valid frame counts for each batch item used to mask padded frames.
            condition (Tensor, optional): Optional global conditioning tensor of shape (B, T, cond_channels).
                A 2D tensor of shape (B, cond_channels) is also accepted and will be broadcast across time.

        Returns:
            Tensor: Output waveform tensor of shape (B, out_feats, T_out).
        """
        x = x.transpose(1, 2).contiguous()  # (B, T, in_feats) -> (B, in_feats, T)
        if condition is not None:
            if condition.dim() == 2:
                condition = condition.unsqueeze(2)
            else:
                assert condition.dim() == 3, "Conditioning tensor must be 2D or 3D"
                condition = condition.transpose(1, 2)

            condition = condition.contiguous()

        if x_lengths is not None:
            if x_lengths.device != x.device:
                raise ValueError("x_lengths must be on the same device as x")
            x_mask = seq_lengths_to_mask(x_lengths, x.size(2), time_dim=2).to(x.dtype)
            x = x * x_mask
        else:
            x_mask = None

        x = self.in_conv(x)
        if x_mask is not None:
            x = x * x_mask

        if condition is not None and self.cond_layer is not None:
            x = x + self.cond_layer(condition)
            if x_mask is not None:
                x = x * x_mask

        for i in range(self.num_upsamples):
            prev_t = x.size(2)
            x = self.activation(x)
            if x_mask is not None:
                x = x * x_mask
            x = self.upsample_layers[i](x)
            if x_lengths is not None:
                x_lengths = scale_seq_lengths(
                    x_lengths, x.size(2), max_in_length=prev_t
                )
                x_mask = seq_lengths_to_mask(
                    x_lengths, x.size(2), time_dim=2, dtype=x.dtype
                )
                x = x * x_mask

            res_out = None
            for j in range(self.num_kernels):
                block = self.blocks[i * self.num_kernels + j]
                x_block = block(x, x_mask=x_mask)
                res_out = x_block if res_out is None else res_out + x_block

            x = res_out / self.num_kernels  # average residuals

        x = self.activation(x)
        if x_mask is not None:
            x = x * x_mask
        x = self.out_conv(x)
        x = torch.tanh(x)
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """
        Removes weight normalization from all layers for efficient inference.
        """
        logging.info("Removing weight norm...")
        for l in self.upsample_layers:
            if is_parametrized(l):
                remove_parametrizations(l, "weight")

        for block in self.blocks:
            block.remove_weight_norm()

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """
        Returns a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the base ``class_name`` entry.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstructing the generator.
        """
        activation = AF.get_config(self.activation)
        config = {
            "in_feats": self.in_feats,
            "out_feats": self.out_feats,
            "resb_kernel_sizes": self.resb_kernel_sizes,
            "resb_dilations": self.resb_dilations,
            "upsample_init_channels": self.upsample_init_channels,
            "upsample_kernel_sizes": self.upsample_kernel_sizes,
            "upsample_strides": self.upsample_strides,
            "activation": activation,
            "cond_channels": self.cond_channels,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: Set[str] = set()
    ) -> None:
        """
        Adds generator arguments to the CLI parser.

        Args:
            parser (ArgumentParser): Argument parser object.
            prefix (str, optional): Optional prefix for argument names.
            skip (Set[str], optional): Argument names to omit from the parser.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats", type=int, default=80, help="Input feature channels."
            )
        if "out_feats" not in skip:
            parser.add_argument(
                "--out-feats",
                type=int,
                default=1,
                help="Output feature channels (e.g., 1 for mono audio).",
            )
        if "resb_kernel_sizes" not in skip:
            parser.add_argument(
                "--resb-kernel-sizes",
                type=int,
                nargs="+",
                default=[3, 7, 11],
                help="Residual block kernel sizes.",
            )
        if "resb_dilations" not in skip:
            parser.add_argument(
                "--resb-dilations",
                type=int,
                nargs="+",
                default=[1, 3, 5],
                help="Dilation rates for residual blocks.",
            )
        if "upsample_init_channels" not in skip:
            parser.add_argument(
                "--upsample-init-channels",
                type=int,
                default=512,
                help="Initial number of channels before upsampling.",
            )
        if "upsample_kernel_sizes" not in skip:
            parser.add_argument(
                "--upsample-kernel-sizes",
                type=int,
                nargs="+",
                default=[16, 16, 4, 4],
                help="Upsample kernel sizes.",
            )
        if "upsample_strides" not in skip:
            parser.add_argument(
                "--upsample-strides",
                type=int,
                nargs="+",
                default=[10, 8, 2, 2],
                help="Upsample stride sizes.",
            )
        if "activation" not in skip:
            parser.add_argument(
                "--activation",
                type=str,
                default="leakyrelu",
                help="Activation function type.",
            )
        if "cond_channels" not in skip:
            parser.add_argument(
                "--cond-channels",
                type=int,
                default=0,
                help="Conditioning input channels.",
            )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

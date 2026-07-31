"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, Conv1d, Linear

from ..layer_blocks import (
    Res2NetBasicBlock,
    Res2NetBNBlock,
    ResNetBasicBlock,
    ResNetBNBlock,
    ResNetEndpointBlock,
    ResNetInputBlock,
    SEResNetBasicBlock,
    SEResNetBNBlock,
)
from ..layers import ActivationFactory as AF
from ..layers import NormLayer2dFactory as NLF
from ..utils import scale_seq_lengths, seq_lengths_to_mask
from .net_arch import NetArch


class ResNet(NetArch):
    """2-D ResNet backbone with optional SE/Res2Net variants and classifier head.

    Attributes:
        block (Union[str, Type[nn.Module]]): Identifier or class describing which residual cell
            to use (``"basic"``, ``"bn"``, ``"sebasic"``, ``"sebn"``, ``"res2basic"``, ``"res2bn"``,
            ``"seres2bn"`` or a custom implementation).
        num_layers (Sequence[int]): Number of residual blocks in each of the four stages.
        in_channels (int): Number of channels expected by the stem convolution.
        conv_channels (int): Channels produced by the stem convolution block.
        base_channels (int): Channels in the first residual stage; later stages scale from this value.
        out_units (int): Size of the optional classification head (``0`` disables the head).
        hid_act (Union[str, Dict[str, Any]]): Hidden activation specification consumed by
            :class:`ActivationFactory`.
        out_act (Optional[Union[str, Dict[str, Any]]]): Output activation for the classification head.
        zero_init_residual (bool): Whether to zero-initialise the last BN inside each block.
        multilevel (bool): Enables multi-resolution endpoint pooling when ``True``.
        endpoint_channels (int): Output channels for each endpoint when ``multilevel`` is enabled.
        groups (int): Group count used by convolutions inside the blocks.
        replace_stride_with_dilation (Sequence[bool]): Flags indicating which stages replace stride with dilation.
        dropout_rate (float): Dropout probability applied inside residual blocks.
        norm_layer (Optional[Union[str, Type[nn.Module], Callable[[int], nn.Module]]]): Normalisation factory used
            throughout the network. Defaults to BatchNorm2d when ``None``.
        norm_before (bool): If ``True`` applies normalisation before activations in residual blocks.
        do_maxpool (bool): Whether to include the max-pooling layer in the stem.
        in_norm (bool): Adds an extra normalisation layer at the input when ``True``.
        se_r (int): Squeeze–excitation reduction ratio.
        se_type (str): Type of squeeze–excitation pooling (e.g. ``"cw-se"``, ``"t-se"``).
        in_feats (Optional[int]): Input feature size used to size squeeze–excitation layers when ``time_se`` is ``True``.
        res2net_scale (int): Res2Net scale hyper-parameter.
        res2net_width_factor (int): Factor scaling the width of Res2Net bottlenecks.
        resb_channels (Optional[Sequence[int]]): Optional list specifying channel sizes per residual stage.
        time_se (bool): Uses time-only pooling for squeeze–excitation when ``True``.
        freq_pos_enc (bool): Enables frequency positional encoding inside squeeze–excitation layers.
    """

    def __init__(
        self,
        block: Union[str, Type[nn.Module]],
        num_layers: Sequence[int],
        in_channels: int,
        conv_channels: int = 64,
        base_channels: int = 64,
        out_units: int = 0,
        hid_act: Union[str, Dict[str, Any]] = {"name": "relu", "inplace": True},
        out_act: Optional[Union[str, Dict[str, Any]]] = None,
        in_kernel_size: int = 7,
        in_stride: int = 2,
        zero_init_residual: bool = False,
        multilevel: bool = False,
        endpoint_channels: int = 64,
        groups: int = 1,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        dropout_rate: float = 0.0,
        norm_layer: Optional[
            Union[str, Type[nn.Module], Callable[[int], nn.Module]]
        ] = None,
        norm_before: bool = True,
        do_maxpool: bool = True,
        in_norm: bool = True,
        se_r: int = 16,
        se_type: str = "cw-se",
        in_feats: Optional[int] = None,
        res2net_scale: int = 4,
        res2net_width_factor: int = 1,
        resb_channels: Optional[Sequence[int]] = None,
        time_se: bool = False,
        freq_pos_enc: bool = False,
    ) -> None:
        """Create a configurable 2-D ResNet backbone.

        Args:
            block: Residual block type or custom block class.
            num_layers: Number of residual blocks in each stage.
            in_channels: Number of input channels.
            conv_channels: Number of channels produced by the stem convolution.
            base_channels: Number of channels in the first residual stage.
            out_units: Size of the optional classification head; ``0`` disables it.
            hid_act: Hidden activation specification.
            out_act: Optional output activation specification.
            in_kernel_size: Kernel size of the stem convolution.
            in_stride: Stride of the stem convolution.
            zero_init_residual: If ``True``, zero-initialize the last BN in each residual branch.
            multilevel: If ``True``, expose intermediate endpoint features.
            endpoint_channels: Output channels for multilevel endpoint projections.
            groups: Number of groups in grouped convolutions.
            replace_stride_with_dilation: Flags that replace stage strides with dilation.
            dropout_rate: Dropout probability inside residual blocks.
            norm_layer: Normalization-layer constructor or alias.
            norm_before: If ``True``, apply normalization before activation.
            do_maxpool: If ``True``, keep the max-pooling layer in the stem.
            in_norm: If ``True``, apply normalization to the input tensor.
            se_r: Squeeze-excitation reduction ratio.
            se_type: Squeeze-excitation pooling type.
            in_feats: Input feature size required by time/frequency SE variants.
            res2net_scale: Res2Net scale factor.
            res2net_width_factor: Width multiplier for Res2Net blocks.
            resb_channels: Optional per-stage residual channel sizes.
            time_se: If ``True``, use time-only squeeze-excitation pooling.
            freq_pos_enc: If ``True``, enable frequency positional encoding in SE blocks.
        """
        super().__init__()
        logging.info("{}".format(locals()))
        self.block: Union[str, Type[nn.Module]] = block
        self.has_se = False
        self.is_res2net = False

        if isinstance(block, str):
            if block == "basic":
                self._block: Type[nn.Module] = ResNetBasicBlock
            elif block == "bn":
                self._block = ResNetBNBlock
            elif block == "sebasic":
                self._block = SEResNetBasicBlock
                self.has_se = True
            elif block == "sebn":
                self._block = SEResNetBNBlock
                self.has_se = True
            elif block == "res2basic":
                self._block = Res2NetBasicBlock
                self.is_res2net = True
            elif block == "res2bn":
                self._block = Res2NetBNBlock
                self.is_res2net = True
            elif block == "seres2basic":
                self._block = Res2NetBasicBlock
                self.has_se = True
                self.is_res2net = True
            elif block in ("seres2bn", "tseres2bn"):
                self._block = Res2NetBNBlock
                self.has_se = True
                self.is_res2net = True
            else:
                raise ValueError(f"Unsupported ResNet block type: {block}")
        else:
            self._block = block
            self.has_se = getattr(block, "has_se", False)
            self.is_res2net = issubclass(block, (Res2NetBasicBlock, Res2NetBNBlock))

        assert not self.has_se and not freq_pos_enc or in_feats is not None

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.base_channels = base_channels
        self.out_units = out_units
        self.in_kernel_size = in_kernel_size
        self.in_stride = in_stride
        self.hid_act = hid_act
        self.groups = groups
        self.norm_before = norm_before
        self.do_maxpool = do_maxpool
        self.in_norm = in_norm
        self.dropout_rate = dropout_rate
        # self.width_per_group = width_per_group
        self.se_r = se_r
        self.time_se = time_se
        if time_se:
            se_type = "t-se"
        self.se_type = se_type
        self.in_feats = in_feats
        self.res2net_scale = res2net_scale
        self.res2net_width_factor = res2net_width_factor
        self.resb_channels = resb_channels

        self.multilevel = multilevel
        self.endpoint_channels = endpoint_channels
        self.freq_pos_enc = freq_pos_enc

        self.norm_layer = norm_layer
        norm_groups = None
        if norm_layer == "group-norm":
            norm_groups = min(base_channels // 2, 32)
            norm_groups = max(norm_groups, groups)
        self._norm_layer: Callable[[int], nn.Module] = NLF.create(
            norm_layer, norm_groups
        )

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.groups = groups
        # self.width_per_group = width_per_group

        self.in_bn: Optional[nn.Module] = None
        if in_norm:
            self.in_bn = self._norm_layer(in_channels)

        self.in_block = ResNetInputBlock(
            in_channels,
            conv_channels,
            kernel_size=in_kernel_size,
            stride=in_stride,
            activation=hid_act,
            norm_layer=self._norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
        )

        self._context = self.in_block.context
        self._downsample_factor = self.in_block.downsample_factor

        if resb_channels is None:
            resb_channels = [base_channels * (2**i) for i in range(4)]

        self.cur_in_channels = conv_channels
        self.layer1 = self._make_layer(self._block, resb_channels[0], num_layers[0])
        self.layer2 = self._make_layer(
            self._block,
            # 2 * base_channels,
            resb_channels[1],
            num_layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            self._block,
            # 4 * base_channels,
            resb_channels[2],
            num_layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            self._block,
            # 8 * base_channels,
            resb_channels[3],
            num_layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        if self.multilevel:
            self.endpoint2 = ResNetEndpointBlock(
                2 * base_channels * self._block.expansion,
                self.endpoint_channels,
                1,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )
            self.endpoint3 = ResNetEndpointBlock(
                4 * base_channels * self._block.expansion,
                self.endpoint_channels,
                2,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )
            self.endpoint4 = ResNetEndpointBlock(
                8 * base_channels * self._block.expansion,
                self.endpoint_channels,
                4,
                activation=self.hid_act,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
            )

        self.with_output = False
        self.out_act = None
        if out_units > 0:
            self.with_output = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output = nn.Linear(self.cur_in_channels, out_units)
            self.out_act = AF.create(out_act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                act_name = "relu"
                if isinstance(hid_act, str):
                    act_name = hid_act
                if isinstance(hid_act, dict):
                    act_name = hid_act["name"]
                if act_name == "swish":
                    act_name = "relu"
                try:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=act_name
                    )
                except:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        self.zero_init_residual = zero_init_residual
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBNBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[nn.Module],
        channels: int,
        num_blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """Build one residual stage.

        Args:
            block: Residual block class used for the stage.
            channels: Output channels for the stage.
            num_blocks: Number of residual blocks to stack.
            stride: Stride for the first block in the stage.
            dilate: If ``True``, replace the first stride with dilation.

        Returns:
            nn.Sequential: The assembled residual stage.
        """
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        kwargs = {}
        if self.has_se:
            if self.se_type == "cw-se":
                kwargs = {"se_r": self.se_r}
            else:
                num_feats = int(self.in_feats / (self._downsample_factor * stride))
                kwargs = {
                    "se_r": self.se_r,
                    "se_type": self.se_type,
                    "num_feats": num_feats,
                }

            if self.freq_pos_enc:
                kwargs["freq_pos_enc"] = True
                num_feats = int(self.in_feats / (self._downsample_factor * stride))
                kwargs["num_feats"] = num_feats

        if self.is_res2net:
            kwargs["scale"] = self.res2net_scale
            kwargs["width_factor"] = self.res2net_width_factor

        layers: List[nn.Module] = []
        layers.append(
            block(
                self.cur_in_channels,
                channels,
                activation=self.hid_act,
                stride=stride,
                dropout_rate=self.dropout_rate,
                groups=self.groups,
                dilation=previous_dilation,
                norm_layer=self._norm_layer,
                norm_before=self.norm_before,
                **kwargs,
            )
        )

        self._context += layers[0].context * self._downsample_factor
        self._downsample_factor *= layers[0].downsample_factor

        self.cur_in_channels = channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    self.cur_in_channels,
                    channels,
                    activation=self.hid_act,
                    dropout_rate=self.dropout_rate,
                    groups=self.groups,
                    dilation=self.dilation,
                    norm_layer=self._norm_layer,
                    norm_before=self.norm_before,
                    **kwargs,
                )
            )

            self._context += layers[-1].context * self._downsample_factor

        return nn.Sequential(*layers)

    def _compute_out_size(self, in_size: int) -> int:
        """Compute the spatial resolution after the ResNet stack.

        Args:
            in_size (int): Spatial size (height or width) of the input tensor.

        Returns:
            int: Spatial size after applying the stem, pooling, and residual strides.
        """
        out_size = int((in_size - 1) // self.in_stride + 1)
        if self.do_maxpool:
            out_size = int((out_size - 1) // 2 + 1)

        for i in range(3):
            if not self.replace_stride_with_dilation[i]:
                out_size = int((out_size - 1) // 2 + 1)

        return out_size

    def _compute_hid_sizes(self, in_size: int, layers: List[int]) -> List[int]:
        """Compute spatial resolutions after intermediate ResNet stages.

        Args:
            in_size (int): Spatial size (height or width) of the input tensor.
            layers (List[int]): Which stages (0=post-stem, 1-3=residual stages) to report.

        Returns:
            List[int]: Spatial sizes for the requested stages, in the same order as ``layers``.
        """
        sizes = {}
        out_size = int((in_size - 1) // self.in_stride + 1)
        if self.do_maxpool:
            out_size = int((out_size - 1) // 2 + 1)

        if 0 in layers:
            sizes[0] = out_size

        if 1 in layers:
            sizes[1] = out_size

        for i in range(3):
            if not self.replace_stride_with_dilation[i]:
                out_size = int((out_size - 1) // 2 + 1)

            if (i + 2) in layers:
                sizes[i + 2] = out_size

        return [sizes[i] for i in layers]

    def in_context(self) -> Tuple[int, int]:
        """Return the receptive-field context ``(past, future)`` in frames.

        Returns:
            Tuple[int, int]: Context needed to predict a frame on the left/right.
        """
        return (self._context, self._context)

    def in_shape(self) -> Tuple[Optional[int], int, Optional[int], Optional[int]]:
        """Describe the expected input shape ``(batch, channels, height, width)``.

        Returns:
            Tuple[Optional[int], int, Optional[int], Optional[int]]: Shape descriptor.
        """
        return (None, self.in_channels, None, None)

    def out_shape(
        self,
        in_shape: Optional[
            Tuple[Optional[int], int, Optional[int], Optional[int]]
        ] = None,
    ) -> Tuple[Any, ...]:
        """Infer the output shape given an input shape.

        Args:
            in_shape (Optional[Tuple[Optional[int], int, Optional[int], Optional[int]]]): Optional tuple describing
                the batch size, channel count, height, and width of the input tensor.

        Returns:
            Tuple[Any, ...]: Output shape consistent with the configured ResNet.
        """

        if self.with_output:
            return (None, self.out_units)

        if in_shape is None:
            if self.multilevel:
                return (None, self.endpoint_channels, None, None)
            else:
                return (None, self.layer4[-1].out_channels, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = None
        else:
            H = self._compute_out_size(in_shape[2])

        if in_shape[3] is None:
            W = None
        else:
            W = self._compute_out_size(in_shape[3])

        if self.multilevel:
            return (in_shape[0], self.endpoint_channels, int(in_shape[2] // 2), None)

        return (in_shape[0], self.layer4[-1].out_channels, H, W)

    def hid_shapes(
        self,
        layers: List[int],
        in_shape: Optional[
            Tuple[Optional[int], int, Optional[int], Optional[int]]
        ] = None,
    ) -> Tuple[Any, ...]:
        """Compute hidden feature-map shapes for the requested layers.

        Args:
            layers (List[int]): Indices of the backbone stages to inspect (0=in_block, 1-4=resnet blocks).
            in_shape (Optional[Tuple[Optional[int], int, Optional[int], Optional[int]]]): Optional tuple describing
                the batch size, channel count, height, and width of the input tensor.

        Returns:
            Tuple[Any, ...]: Per-layer shapes matching the order of ``layers``.
        """
        shapes = []
        if in_shape is None:
            in_shape = (None, None, None, None)

        assert len(in_shape) == 4
        if in_shape[2] is None:
            H = [None] * len(layers)
        else:
            H = self._compute_hid_sizes(in_shape[2], layers)

        if in_shape[3] is None:
            W = [None] * len(layers)
        else:
            W = self._compute_hid_sizes(in_shape[3], layers)

        C = (
            self.in_block.out_channels,
            self.layer1[-1].out_channels,
            self.layer2[-1].out_channels,
            self.layer3[-1].out_channels,
            self.layer4[-1].out_channels,
        )
        C = [C[i] for i in layers]
        shapes = [(in_shape[0], C[i], H[i], W[i]) for i in range(len(layers))]
        return shapes

    @staticmethod
    def _forward_layer_with_lens(
        layer: nn.Sequential,
        x: torch.Tensor,
        in_lengths: torch.Tensor,
        max_in_length: int,
    ) -> torch.Tensor:
        """Forward a stage while tracking valid sequence lengths.

        Args:
            layer: Residual stage to execute.
            x: Input feature map.
            in_lengths: Valid lengths for the original input tensor.
            max_in_length: Original unpadded temporal length before resizing.

        Returns:
            torch.Tensor: Output feature map after masking padded positions.
        """
        x_lengths = scale_seq_lengths(in_lengths, x.size(-1), max_in_length)
        x_mask = seq_lengths_to_mask(x_lengths, x.size(-1), time_dim=3, dtype=x.dtype)

        for sub_layer in layer:
            if sub_layer.stride > 1:
                x_mask = x_mask[..., :: sub_layer.stride]

            x = sub_layer(x, x_mask)

        return x

    @staticmethod
    def _forward_layer_with_mask(
        layer: nn.Sequential, x: torch.Tensor, x_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward a stage using an existing validity mask.

        Args:
            layer: Residual stage to execute.
            x: Input feature map.
            x_mask: Mask marking valid temporal positions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated feature map and mask.
        """

        for sub_layer in layer:
            if sub_layer.stride > 1:
                x_mask = x_mask[..., :: sub_layer.stride]

            x = sub_layer(x, x_mask)

        return x, x_mask

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the ResNet forward pass and return logits or feature maps.

        Args:
            x (torch.Tensor): Input tensor shaped `(batch, channels, height, width)` or `(batch, channels, freq, time)`.
            x_lengths (Optional[torch.Tensor]): Optional sequence lengths aligned with the last dimension.

        Returns:
            torch.Tensor: Logits `(batch, out_units)` when ``out_units > 0``; otherwise feature maps.
        """
        if x_lengths is not None:
            # if all lengths are eq. to the max length, we set x_lengths to None
            max_length = x.size(-1)
            if torch.all(x_lengths == max_length):
                x_lengths = None

        if self.in_norm and self.in_bn is not None:
            x = self.in_bn(x)
        feats = []
        x = self.in_block(x)

        if x_lengths is None:
            x = self.layer1(x)
            x = self.layer2(x)
            if self.multilevel:
                feats.append(x)
            x = self.layer3(x)
            if self.multilevel:
                feats.append(x)
            x = self.layer4(x)
            if self.multilevel:
                feats.append(x)
        else:
            if max_length != x.size(-1):
                x_lengths = scale_seq_lengths(x_lengths, x.size(-1), max_length)
            x_mask = seq_lengths_to_mask(
                x_lengths, x.size(-1), time_dim=3, dtype=x.dtype
            )
            x, x_mask = self._forward_layer_with_mask(self.layer1, x, x_mask)
            x, x_mask = self._forward_layer_with_mask(self.layer2, x, x_mask)
            if self.multilevel:
                feats.append(x)
            x, x_mask = self._forward_layer_with_mask(self.layer3, x, x_mask)
            if self.multilevel:
                feats.append(x)
            x, x_mask = self._forward_layer_with_mask(self.layer4, x, x_mask)
            if self.multilevel:
                feats.append(x)

        if self.multilevel:
            out2 = self.endpoint2(feats[0])
            out3 = self.endpoint3(feats[1])
            out4 = self.endpoint4(feats[2])
            x = torch.mean(torch.stack([out2, out3, out4]), 0)

        if self.with_output:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.output(x)
            if self.out_act is not None:
                x = self.out_act(x)

        return x

    def forward_hid_feats(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None,
        return_output: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        """Return intermediate activations (and optionally the final output).

        Args:
            x (torch.Tensor): Input tensor shaped ``(batch, channels, height, width)`` or `(batch, channels, freq, time)`.
            x_lengths (Optional[torch.Tensor]): Optional sequence lengths aligned with the time dimension.
            layers (Optional[List[int]]): Indices of intermediate stages whose activations should be collected.
            return_output (bool): If ``True`` also return the final output tensor.

        Returns:
            Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
                Hidden activations and, when ``return_output`` is ``True``, a tuple containing
                both the activations and the final output tensor.
        """
        assert layers is not None or return_output
        if layers is None:
            layers = []
        if len(layers) == 0 and not return_output:
            return []

        if return_output:
            last_layer = 4
        else:
            last_layer = max(layers)

        h = []
        feats = []
        max_length = x.size(-1)
        if x_lengths is not None and torch.all(x_lengths == max_length):
            x_lengths = None

        if self.in_norm and self.in_bn is not None:
            x = self.in_bn(x)

        x = self.in_block(x)
        if 0 in layers:
            h.append(x)
        if last_layer == 0:
            return h

        if x_lengths is None:
            x = self.layer1(x)
            if 1 in layers:
                h.append(x)
            if last_layer == 1:
                return h

            x = self.layer2(x)
            if 2 in layers:
                h.append(x)
            if last_layer == 2:
                return h
            if return_output and self.multilevel:
                feats.append(x)

            x = self.layer3(x)
            if 3 in layers:
                h.append(x)
            if last_layer == 3:
                return h
            if return_output and self.multilevel:
                feats.append(x)

            x = self.layer4(x)
            if 4 in layers:
                h.append(x)
            if return_output and self.multilevel:
                feats.append(x)
        else:
            if max_length != x.size(-1):
                x_lengths = scale_seq_lengths(x_lengths, x.size(-1), max_length)
            x_mask = seq_lengths_to_mask(
                x_lengths, x.size(-1), time_dim=3, dtype=x.dtype
            )

            x, x_mask = self._forward_layer_with_mask(self.layer1, x, x_mask)
            if 1 in layers:
                h.append(x)
            if last_layer == 1:
                return h

            x, x_mask = self._forward_layer_with_mask(self.layer2, x, x_mask)
            if 2 in layers:
                h.append(x)
            if last_layer == 2:
                return h
            if return_output and self.multilevel:
                feats.append(x)

            x, x_mask = self._forward_layer_with_mask(self.layer3, x, x_mask)
            if 3 in layers:
                h.append(x)
            if last_layer == 3:
                return h
            if return_output and self.multilevel:
                feats.append(x)

            x, x_mask = self._forward_layer_with_mask(self.layer4, x, x_mask)
            if 4 in layers:
                h.append(x)
            if return_output and self.multilevel:
                feats.append(x)

        if return_output:
            if self.multilevel:
                out2 = self.endpoint2(feats[0])
                out3 = self.endpoint3(feats[1])
                out4 = self.endpoint4(feats[2])
                x = torch.mean(torch.stack([out2, out3, out4]), 0)

            return h, x

        return h

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the constructor arguments.

        Returns:
            Dict[str, Any]: Configuration dictionary that can be fed back into ``__init__``.
        """

        out_act = AF.get_config(self.out_act)
        hid_act = self.hid_act

        config = {
            "block": self.block,
            "num_layers": self.num_layers,
            "in_channels": self.in_channels,
            "conv_channels": self.conv_channels,
            "base_channels": self.base_channels,
            "out_units": self.out_units,
            "in_kernel_size": self.in_kernel_size,
            "in_stride": self.in_stride,
            "zero_init_residual": self.zero_init_residual,
            "groups": self.groups,
            "replace_stride_with_dilation": self.replace_stride_with_dilation,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "norm_before": self.norm_before,
            "in_norm": self.in_norm,
            "do_maxpool": self.do_maxpool,
            "out_act": out_act,
            "hid_act": hid_act,
            "se_r": self.se_r,
            "se_type": self.se_type,
            "in_feats": self.in_feats,
            "res2net_scale": self.res2net_scale,
            "res2net_width_factor": self.res2net_width_factor,
            "resb_channels": self.resb_channels,
            "freq_pos_enc": self.freq_pos_enc,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))


# Standard ResNets
class ResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("basic", [2, 2, 2, 2], in_channels, **kwargs)


class ResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)


class ResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class ResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class ResNet152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("bn", [3, 8, 36, 3], in_channels, **kwargs)


class ResNext50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class ResNext101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class WideResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class WideResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        super().__init__("bn", [3, 4, 23, 3], in_channels, **kwargs)


class IdRndResNet100(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        super().__init__("basic", [6, 16, 24, 3], in_channels, **kwargs)


class IdRndResNet202(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        super().__init__("basic", [6, 16, 75, 3], in_channels, **kwargs)


class LResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("basic", [2, 2, 2, 2], in_channels, **kwargs)


class LResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)


class LResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


class LResNext50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("bn", [3, 4, 6, 3], in_channels, **kwargs)


# multi-level feature ResNet
class LResNet34_345(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["multilevel"] = True
        kwargs["endpoint_channels"] = 64
        super().__init__("basic", [3, 4, 6, 3], in_channels, **kwargs)


# Squezee-Excitation ResNets


class SEResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class SEResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SEResNet152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class SEResNext50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEResNext101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SEWideResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SEWideResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["base_channels"] = 128
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class SELResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class SELResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class SELResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class SELResNext50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


# Time dimension Squezee-Excitation ResNets


class TSEResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class TSEResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEResNet152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class TSEResNext50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEResNext101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEWideResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEWideResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class TSELResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class TSELResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class TSELResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class TSELResNext50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


# Freq-wise Squezee-Excitation ResNets


class FwSEResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class FwSEResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSEResNet152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class FwSEResNext50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEResNext101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSEWideResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEWideResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSELResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class FwSELResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class FwSELResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSELResNext50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEIdRndResNet100(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [6, 16, 24, 3], in_channels, **kwargs)


class FwSEIdRndResNet202(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        kwargs["se_type"] = "fw-se"
        super().__init__("sebasic", [6, 16, 75, 3], in_channels, **kwargs)


# Channel-Freq-wise Squezee-Excitation ResNets


class CFwSEResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class CFwSEResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSEResNet152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 8, 36, 3], in_channels, **kwargs)


class CFwSEResNext50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEResNext101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSEWideResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEWideResNet101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSELResNet18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [2, 2, 2, 2], in_channels, **kwargs)


class CFwSELResNet34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSELResNet50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSELResNext50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEIdRndResNet100(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [6, 16, 24, 3], in_channels, **kwargs)


class CFwSEIdRndResNet202(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["resb_channels"] = [128, 128, 256, 256]
        kwargs["se_type"] = "cfw-se"
        super().__init__("sebasic", [6, 16, 75, 3], in_channels, **kwargs)


#################### Res2Net variants ########################


# Standard Res2Nets
class Res2Net18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("res2basic", [2, 2, 2, 2], in_channels, **kwargs)


class Res2Net34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("res2basic", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class Res2Net152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("res2bn", [3, 8, 36, 3], in_channels, **kwargs)


class Res2Next50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class Res2Next101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class WideRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class WideRes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        super().__init__("res2bn", [3, 4, 23, 3], in_channels, **kwargs)


class LRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


class LRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("res2bn", [3, 4, 6, 3], in_channels, **kwargs)


# Squezee-Excitation Res2Nets
class SERes2Net18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("seres2basic", [2, 2, 2, 2], in_channels, **kwargs)


class SERes2Net34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("seres2basic", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SERes2Net152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class SERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SEWideRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SEWideRes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class SELRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class SELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


# Time dimension Squezee-Excitation Res2Nets
class TSERes2Net18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("seres2basic", [2, 2, 2, 2], in_channels, **kwargs)


class TSERes2Net34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("seres2basic", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSERes2Net152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class TSERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSEWideRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSEWideRes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class TSELRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class TSELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["time_se"] = True
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


# frequency-wise  Squezee-Excitation Res2Nets
class FwSERes2Net18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2basic", [2, 2, 2, 2], in_channels, **kwargs)


class FwSERes2Net34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2basic", [3, 4, 6, 3], in_channels, **kwargs)


class FwSERes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSERes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSERes2Net152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class FwSERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSEWideRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSEWideRes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class FwSELRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class FwSELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "fw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


# channel-frequency-wise  Squezee-Excitation Res2Nets
class CFwSERes2Net18(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2basic", [2, 2, 2, 2], in_channels, **kwargs)


class CFwSERes2Net34(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2basic", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSERes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSERes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSERes2Net152(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 8, 36, 3], in_channels, **kwargs)


class CFwSERes2Next50_32x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSERes2Next101_32x8d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 32
        kwargs["base_channels"] = 256
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSEWideRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSEWideRes2Net101(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["base_channels"] = 128
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 23, 3], in_channels, **kwargs)


class CFwSELRes2Net50(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["conv_channels"] = 16
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)


class CFwSELRes2Next50_4x4d(ResNet):
    def __init__(self, in_channels: int, **kwargs: Any) -> None:
        """Initialize this ResNet variant.

        Args:
            in_channels: Number of input channels.
            **kwargs: Additional ResNet constructor keyword arguments.
        """
        kwargs["groups"] = 4
        kwargs["base_channels"] = 16
        kwargs["se_type"] = "cfw-se"
        super().__init__("seres2bn", [3, 4, 6, 3], in_channels, **kwargs)

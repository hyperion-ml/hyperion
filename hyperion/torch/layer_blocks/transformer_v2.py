"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn as nn
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

from ..layers import ActivationFactory as AF
from ..layers import DropPath1d, GRN1d, Interpolate, RMSNorm
from ..layers.attention_v2 import (
    HFFlashScaledDotProdAttV2,
    ScaledDotProdAttV2,
    SDPBackendType,
    TorchScaledDotProdAttV2,
)
from ..layers.pos_encoder import RotaryPosEncoder
from ..utils import scale_seq_lengths, seq_lengths_to_mask


class TransformerEncoderV2StemType(str, Enum):
    CONV1D = "conv1d"
    CONV2D = "conv2d"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of supported stem block identifiers."""
        return [o.value for o in TransformerEncoderV2StemType]

    @staticmethod
    def to_class(value: "TransformerEncoderV2StemType") -> Type[nn.Module]:
        """Map a stem type identifier to the corresponding implementation."""
        # stem block
        if value == TransformerEncoderV2StemType.CONV1D:
            stem_class = TransfomerV2Conv1dStemBlock
        elif value == TransformerEncoderV2StemType.CONV2D:
            stem_class = TransfomerV2Conv2dStemBlock
        else:
            raise ValueError(f"invalid {value=}")

        return stem_class


class TransformerV2NormLayerType(str, Enum):
    LAYERNORM = "layer-norm"
    RMSNORM = "rms-norm"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of supported normalization layer identifiers."""
        return [o.value for o in TransformerV2NormLayerType]

    @staticmethod
    def to_class(value: Optional["TransformerV2NormLayerType"]) -> Type[nn.Module]:
        """Map a normalization identifier to the corresponding module class."""
        if value is None or value == TransformerV2NormLayerType.LAYERNORM:
            return nn.LayerNorm
        elif value == TransformerV2NormLayerType.RMSNORM:
            return RMSNorm
        else:
            raise ValueError(f"invalid {value=}")


class TransformerV2AttType(str, Enum):
    SDP = "sdp"
    TORCH_SDP = "torch_sdp"
    HF_FLASH_SDP = "hf_flash_sdp"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of supported attention backend identifiers."""
        return [o.value for o in TransformerV2AttType]

    @staticmethod
    def to_class(value: "TransformerV2AttType") -> Type[ScaledDotProdAttV2]:
        """Map an attention identifier to the concrete attention module."""
        if value == TransformerV2AttType.SDP:
            return ScaledDotProdAttV2
        elif value == TransformerV2AttType.TORCH_SDP:
            return TorchScaledDotProdAttV2
        elif value == TransformerV2AttType.HF_FLASH_SDP:
            return HFFlashScaledDotProdAttV2
        else:
            raise ValueError(f"invalid {value=}")


class TransformerV2FeedForwardType(str, Enum):
    MLP = "mlp"
    CONVNEXT = "convnext"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of supported feed-forward block identifiers."""
        return [o.value for o in TransformerV2FeedForwardType]

    @staticmethod
    def to_class(value: "TransformerV2FeedForwardType") -> Type[nn.Module]:
        """Map a feed-forward identifier to the corresponding block class."""
        if value == TransformerV2FeedForwardType.MLP:
            return TransformerV2MLPBlock
        elif value == TransformerV2FeedForwardType.CONVNEXT:
            return TransformerV2ConvNextBlock
        else:
            raise ValueError(f"invalid {value=}")


class Conv2dStemLayer(nn.Module):
    """Two-dimensional convolutional stem used by transformer front-ends.

    Attributes:
        conv (nn.Conv2d): Convolution applied to the input spectrogram frames.
        norm (nn.Module): Normalization layer applied channel-wise after the convolution.
        act (nn.Module): Activation function applied after normalization.
        context (int): Effective look-back/look-ahead context introduced by the convolution.
        stride (int): Downsampling factor applied along the time axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str,
        norm_layer: Type[nn.Module],
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        """Initialize the 2-D convolutional stem layer.

        Args:
            in_channels (int): Number of channels expected in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the temporal kernel; automatically clamped to at least ``stride``.
            stride (int): Temporal stride applied by the convolution.
            activation (str): Name of the activation function created through :class:`ActivationFactory`.
            norm_layer (Type[nn.Module]): Normalization layer constructor applied after the convolution.
            bias (bool, optional): Whether to include a bias term in the convolution. Defaults to ``True``.
            norm_eps (float, optional): Epsilon passed to the normalization layer. Defaults to ``1e-5``.
        """
        super().__init__()

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=(2, stride),
            padding=(0, padding),
            bias=bias,
        )
        self.norm = norm_layer(out_channels, eps=norm_eps)
        self.act = AF.create(activation)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution → normalization → activation to 2-D inputs.

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, channels, freq, time)``.

        Returns:
            torch.Tensor: Tensor with the same shape as ``x`` after convolution, normalization, and activation.
        """
        x = self.conv(x)
        x = self.act(self.norm(x.permute(0, 2, 3, 1)))
        return x.permute(0, 3, 1, 2)  # .contiguous()


class Conv1dStemLayer(nn.Module):
    """One-dimensional convolutional stem for sequence inputs.

    Attributes:
        conv (nn.Conv1d): Convolution applied along the temporal dimension.
        norm (nn.Module): Normalization layer applied after convolution.
        act (nn.Module): Activation function applied after normalization.
        context (int): Effective receptive field introduced by the convolution.
        stride (int): Temporal downsampling factor applied by the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        activation: str,
        norm_layer: Type[nn.Module],
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        """Initialize the 1-D convolutional stem layer.

        Args:
            in_channels (int): Number of channels expected in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Temporal kernel size; automatically clamped to at least ``stride``.
            stride (int): Temporal stride applied by the convolution.
            activation (str): Name of the activation function created through :class:`ActivationFactory`.
            norm_layer (Type[nn.Module]): Normalization layer constructor applied after the convolution.
            bias (bool, optional): Whether to include a bias term in the convolution. Defaults to ``True``.
            norm_eps (float, optional): Epsilon passed to the normalization layer. Defaults to ``1e-5``.
        """
        super().__init__()

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = norm_layer(out_channels, eps=norm_eps)
        self.act = AF.create(activation)
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution → normalization → activation to 1-D inputs.

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, channels, time)``.

        Returns:
            torch.Tensor: Tensor with the same shape as ``x`` after convolution, normalization, and activation.
        """
        x = self.conv(x)
        x = self.act(self.norm(x.permute(0, 2, 1)))
        return x.permute(0, 2, 1)  # .contiguous()


class TransfomerV2Conv2dStemBlock(nn.Module):
    """ConvNeXt-V2 inspired stem for 2-D feature inputs.

    Attributes:
        conv_layers (nn.Sequential): Stack of convolutional stem layers.
        norm_layer (nn.Module): Normalization applied after flattening spatial dimensions.
        projection (nn.Linear): Linear projection mapping flattened features to ``out_feats``.
        dropout (nn.Dropout): Dropout applied to the projected features.
        context (int): Total receptive field introduced by the stem.
        downsample_factor (int): Overall temporal downsampling factor produced by the stem.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_channels: List[int] = [128],
        kernel_sizes: List[int] = [4],
        strides: List[int] = [2],
        activation: str = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        norm_eps: float = 1e-5,
        dropout_rate: float = 0.1,
    ):
        """Construct the multi-layer 2-D convolutional stem.

        Args:
            in_feats (int): Incoming feature dimension per frame.
            out_feats (int): Output feature dimension after projection.
            hidden_channels (List[int], optional): Channel widths for each intermediate convolution.
            kernel_sizes (List[int], optional): Kernel sizes for each convolutional layer.
            strides (List[int], optional): Temporal strides for each convolutional layer.
            activation (str, optional): Activation identifier passed to :class:`ActivationFactory`. Defaults to ``"silu"``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
            norm_eps (float, optional): Epsilon provided to the normalization layers. Defaults to ``1e-5``.
            dropout_rate (float, optional): Dropout probability applied after the output projection. Defaults to ``0.1``.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        if norm_layer == RMSNorm:
            conv_bias = True
        else:
            conv_bias = False

        conv_i = Conv2dStemLayer(
            1,
            hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            activation=activation,
            norm_layer=norm_layer,
            bias=conv_bias,
            norm_eps=norm_eps,
        )
        conv_layers = [conv_i]
        feat_dim = in_feats
        # feat_dim = (feat_dim + strides[0] - 1) // strides[0]
        feat_dim = (feat_dim - kernel_sizes[0]) // 2 + 1

        self.context = conv_i.context
        self.downsample_factor = strides[0]
        for i in range(1, len(hidden_channels)):
            conv_i = Conv2dStemLayer(
                hidden_channels[i - 1],
                hidden_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                activation=activation,
                norm_layer=norm_layer,
                bias=conv_bias,
                norm_eps=norm_eps,
            )
            conv_layers.append(conv_i)
            # feat_dim = (feat_dim + strides[i] - 1) // strides[i]
            feat_dim = (feat_dim - kernel_sizes[i]) // 2 + 1
            self.context += conv_i.context * self.downsample_factor
            self.downsample_factor *= strides[i]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.norm_layer = norm_layer(feat_dim * hidden_channels[-1], eps=norm_eps)
        self.projection = nn.Linear(feat_dim * hidden_channels[-1], out_feats)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode 2-D features and return normalized and projected outputs.

        Args:
            x (torch.Tensor): Input tensor shaped ``(batch, time, features)``.
            x_lengths (Optional[torch.Tensor]): Valid lengths for each sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Tuple containing the stem output,
            the projected features, and the updated sequence lengths.
        """
        bs, t_in, f_in = x.size()
        x = x.view(bs, 1, t_in, f_in).permute(0, 1, 3, 2).contiguous()
        x = self.conv_layers(x)
        bs, c, f_out, t_out = x.size()
        x = x.permute(0, 3, 1, 2).reshape(bs, t_out, -1)
        x = self.norm_layer(x)
        if x_lengths is not None:
            x_lengths = scale_seq_lengths(x_lengths, t_out, t_in)
            x_mask = ~seq_lengths_to_mask(x_lengths, t_out)
            x = x.masked_fill(x_mask, 0.0)

        x_proj = self.projection(x)
        x_proj = self.dropout(x_proj)
        if x_lengths is not None:
            x_proj = x.masked_fill(x_mask, 0.0)

        return x, x_proj, x_lengths


class TransfomerV2Conv1dStemBlock(nn.Module):
    """One-dimensional convolutional stem for waveform or feature sequences.

    Attributes:
        conv_layers (nn.Sequential): Stack of convolutional stem layers.
        norm_layer (nn.Module): Normalization applied after the stem convolutions.
        projection (nn.Linear): Linear projection mapping features to ``out_feats``.
        dropout (nn.Dropout): Dropout applied to the projected features.
        context (int): Total receptive field introduced by the stem.
        downsample_factor (int): Recorded cumulative downsampling factor of the stem.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        hidden_channels: List[int] = [128],
        kernel_sizes: List[int] = [4],
        strides: List[int] = [2],
        activation: str = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        norm_eps: float = 1e-5,
        dropout_rate: float = 0.1,
    ):
        """Construct the multi-layer 1-D convolutional stem.

        Args:
            in_feats (int): Number of input channels in the sequence.
            out_feats (int): Number of channels produced by the final projection.
            hidden_channels (List[int], optional): Channel widths for each intermediate convolution.
            kernel_sizes (List[int], optional): Kernel sizes for the convolutional layers.
            strides (List[int], optional): Strides applied by each convolutional layer.
            activation (str, optional): Activation identifier forwarded to :class:`ActivationFactory`. Defaults to ``"silu"``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
            norm_eps (float, optional): Epsilon provided to the normalization layers. Defaults to ``1e-5``.
            dropout_rate (float, optional): Dropout probability applied after the output projection. Defaults to ``0.1``.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        if norm_layer == RMSNorm:
            conv_bias = True
        else:
            conv_bias = False

        conv_i = Conv1dStemLayer(
            in_feats,
            hidden_channels[0],
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            activation=activation,
            norm_layer=norm_layer,
            bias=conv_bias,
            norm_eps=norm_eps,
        )
        conv_layers = [conv_i]

        self.context = conv_i.context
        self.downsample_factor = strides[0]
        for i in range(1, len(hidden_channels)):
            conv_i = Conv1dStemLayer(
                hidden_channels[i - 1],
                hidden_channels[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                activation=activation,
                norm_layer=norm_layer,
                bias=conv_bias,
                norm_eps=norm_eps,
            )
            conv_layers.append(conv_i)
            self.context += conv_i.context * self.downsample_factor
            self.downsample_factor *= strides[i]

        self.conv_layers = nn.Sequential(conv_layers)
        self.norm_layer = norm_layer(hidden_channels[-1], eps=norm_eps)
        self.projection = nn.Linear(hidden_channels[-1], out_feats)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Encode 1-D features and return normalized and projected outputs.

        Args:
            x (torch.Tensor): Input tensor shaped `(batch, time, channels)`.
            x_lengths (Optional[torch.Tensor]): Valid lengths for each sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: Tuple containing the stem output,
            the projected features, and the updated sequence lengths.
        """
        bs, t_in, c = x.size()
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm_layer(x)
        if x_lengths is not None:
            x_lengths = scale_seq_lengths(x_lengths, x.size(1), t_in)
            x_mask = ~seq_lengths_to_mask(x_lengths, x.size(1))
            x = x.masked_fill(x_mask, 0.0)

        x_proj = self.projection(x)
        x_proj = self.dropout(x_proj)
        if x_lengths is not None:
            x_proj = x.masked_fill(x_mask, 0.0)

        return x, x_proj, x_lengths


class TransformerV2MLPBlock(nn.Module):
    """Gated feed-forward network used within the transformer blocks.

    Attributes:
        gate_proj (nn.Module): Projection whose output is gated by the activation.
        up_proj (nn.Module): Parallel projection combined with ``gate_proj`` to build the gated product.
        down_proj (nn.Module): Projection returning activations to ``hidden_dim``.
        act (nn.Module): Activation applied to the gated branch.
        context (int): Effective receptive field size (``1`` for point-wise operations).
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: Union[str, nn.Module] = "silu",
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        model_parallel: bool = False,
        **kwargs,
    ):
        """Initialize the gated MLP block.

        Args:
            hidden_dim (int): Dimension of the incoming feature vectors.
            intermediate_dim (int): Target width of the intermediate projections before rounding.
            activation (Union[str, nn.Module], optional): Activation applied to the gated branch. Defaults to ``"silu"``.
            ff_bias (bool, optional): Whether linear layers include bias terms. Defaults to ``False``.
            ff_multiple_of (int, optional): Rounds ``intermediate_dim`` up to the nearest multiple. Defaults to ``256``.
            model_parallel (bool, optional): If ``True``, uses tensor model-parallel linear layers. Defaults to ``False``.
            **kwargs: Ignored keyword arguments kept for API compatibility.
        """
        super().__init__()
        # mimics LLama 3 readjustemnt of intermediate_dim
        intermediate_dim = ff_multiple_of * (
            (intermediate_dim + ff_multiple_of - 1) // ff_multiple_of
        )

        if model_parallel:
            self.gate_proj = ColumnParallelLinear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
                gather_output=False,
            )
            self.up_proj = RowParallelLinear(
                hidden_dim,
                intermediate_dim,
                bias=False,
                input_is_parallel=True,
            )
            self.down_proj = ColumnParallelLinear(
                intermediate_dim,
                hidden_dim,
                bias=False,
                gather_output=False,
            )
        else:
            self.gate_proj = nn.Linear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
            )
            self.up_proj = nn.Linear(
                hidden_dim,
                intermediate_dim,
                bias=ff_bias,
            )
            self.down_proj = nn.Linear(
                intermediate_dim,
                hidden_dim,
                bias=ff_bias,
            )

        self.act = AF.create(activation)
        self.context = 1

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply the gated MLP to the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, time, hidden_dim)`.
            x_mask (Optional[torch.Tensor]): Unused placeholder for API symmetry.

        Returns:
            torch.Tensor: Projected tensor with shape `(batch, time, hidden_dim)`.
        """
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class TransformerV2ConvNextBlock(nn.Module):
    """ConvNeXt-V2 style depthwise block used as a transformer feed-forward module.

    Attributes:
        dwconv (nn.Conv1d): Depthwise convolution applied along the temporal dimension.
        norm (nn.Module): Normalization layer applied after the depthwise convolution.
        gate_proj (nn.Linear): Projection used in the gated branch.
        up_proj (nn.Linear): Projection combined with ``gate_proj`` to form the gated activations.
        down_proj (nn.Linear): Projection returning activations to ``hidden_dim``.
        act (nn.Module): Activation function applied to the gated branch.
        grn (GRN1d): Global response normalization applied to the intermediate representation.
        context (int): Effective receptive field contributed by the depthwise convolution.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        kernel_size: int = 7,
        dilation: int = 1,
        activation: Union[str, nn.Module] = "silu",
        norm_layer: Optional[Type[nn.Module]] = None,
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        model_parallel: bool = False,
    ):
        """Initialize the ConvNeXt-style feed-forward block.

        Args:
            hidden_dim (int): Dimension of the incoming sequence representation.
            intermediate_dim (int): Target width of the intermediate projections before rounding.
            kernel_size (int, optional): Depthwise convolution kernel size. Defaults to ``7``.
            dilation (int, optional): Dilation applied to the depthwise kernel. Defaults to ``1``.
            activation (Union[str, nn.Module], optional): Activation applied to the gated branch. Defaults to ``"silu"``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
            ff_bias (bool, optional): Whether linear projections include bias terms. Defaults to ``False``.
            ff_multiple_of (int, optional): Rounds ``intermediate_dim`` up to the nearest multiple. Defaults to ``256``.
            model_parallel (bool, optional): Placeholder for parity with :class:`TransformerV2MLPBlock`; must be ``False``.
        """
        super().__init__()
        assert model_parallel is False
        # mimics LLama 3 readjustemnt of intermediate_dim
        intermediate_dim = ff_multiple_of * (
            (intermediate_dim + ff_multiple_of - 1) // ff_multiple_of
        )

        padding = dilation * (kernel_size - 1) // 2
        self.dwconv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=hidden_dim,
        )  # depthwise conv
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.norm = norm_layer(hidden_dim, eps=1e-6)
        self.gate_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=ff_bias,
        )
        self.up_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=ff_bias,
        )
        self.down_proj = nn.Linear(
            intermediate_dim,
            hidden_dim,
            bias=ff_bias,
        )
        self.act = AF.create(activation)
        self.grn = GRN1d(intermediate_dim, channels_last=True)
        self.context = padding

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply ConvNeXt-style depthwise convolution and gated projection.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, time, hidden_dim)``.
            x_mask (Optional[torch.Tensor]): Optional mask passed to :class:`GRN1d`.

        Returns:
            torch.Tensor: Tensor with shape ``(batch, time, hidden_dim)`` after the ConvNeXt transformation.
        """
        # input = x
        x = x.permute(0, 2, 1).contiguous()  # (N, T, C) -> (N, C, T)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.act(self.gate_proj(x)) * self.up_proj(x)
        x = self.grn(x, x_mask)
        x = self.down_proj(x)
        return x


class TransformerV2ConvEndpoint(nn.Module):
    """Resample transformer features to a shared temporal scale for aggregation.

    Attributes:
        in_channels (int): Number of channels received from the transformer layer.
        out_channels (int): Number of channels produced after resampling.
        rel_scale (float): Ratio between input and output temporal resolutions.
        norm (nn.Module): Normalization layer applied before resampling.
        resample (nn.Module): Module performing the up/down-sampling operation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_scale: int,
        out_scale: int,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):

        """Create the resampling endpoint used for multiscale aggregation.

        Args:
            in_channels (int): Number of channels provided by the transformer layer.
            out_channels (int): Number of channels produced after resampling.
            in_scale (int): Temporal resolution of the incoming features.
            out_scale (int): Target temporal resolution for the resampled features.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rel_scale = in_scale / out_scale
        self.norm = norm_layer(in_channels, eps=1e-6)
        if out_scale >= in_scale:
            stride = int(out_scale / in_scale)
            self.resample = self._make_downsample(in_channels, out_channels, stride)
        else:
            stride = int(in_scale / out_scale)
            self.resample = self._make_upsample(
                in_channels,
                out_channels,
                stride,
            )

    @staticmethod
    def _make_downsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:

        if stride % 2 == 0:
            first_stride = 2
            second_stride = stride // 2
        else:
            first_stride = 1
            second_stride = stride

        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=first_stride,
                stride=first_stride,
                bias=True,
            )
        ]

        if second_stride > 1:
            kernel_size = 2 * (second_stride // 2) + 1
            layers.append(
                nn.MaxPool1d(
                    kernel_size=kernel_size,
                    stride=second_stride,
                    padding=(kernel_size - 1) // 2,
                )
            )

        return nn.Sequential(*layers)

    @staticmethod
    def _make_upsample(
        in_channels: int, out_channels: int, stride: int
    ) -> nn.Sequential:
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        ]
        layers.append(Interpolate(scale_factor=stride, mode="nearest"))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resample the input sequence to the target temporal scale.

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, time, in_channels)``.
            x_mask (Optional[torch.Tensor]): Unused; present for API symmetry.

        Returns:
            torch.Tensor: Tensor with shape ``(batch, out_time, out_channels)``.
        """
        x = self.norm(x).permute(0, 2, 1).contiguous()
        x = self.resample(x).permute(0, 2, 1).contiguous()
        return x


class TransformerV2ConvDownsampleBlock(nn.Module):
    """ConvNeXt-V2 style downsampling block for temporal features.

    Attributes:
        norm (nn.Module): Normalization layer applied before downsampling.
        conv (nn.Conv1d): Convolution performing the strided downsampling.
        context (int): Additional temporal context introduced by the convolution.
        stride (int): Downsampling factor applied along the time axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        norm_layer: Optional[Type[nn.Module]] = None,
    ):
        """Initialize the downsampling block.

        Args:
            in_channels (int): Number of input channels before downsampling.
            out_channels (int): Number of channels produced after the strided convolution.
            kernel_size (int, optional): Convolution kernel size; at least ``stride``. Defaults to ``2``.
            stride (int, optional): Temporal stride applied by the convolution. Defaults to ``2``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        kernel_size = max(kernel_size, stride)
        padding = (kernel_size - 1) // 2
        self.norm = norm_layer(in_channels, eps=1e-6)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.context = (kernel_size - 1) // 2
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the temporal resolution via grouped convolution.

        Args:
            x (torch.Tensor): Input tensor with shape ``(batch, time, channels)``.

        Returns:
            torch.Tensor: Downsampled tensor with shape ``(batch, ceil(time / stride), out_channels)``.
        """
        x = self.norm(x)
        return self.conv(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()


class TransformerV2SelfAttBlock(nn.Module):
    """Transformer block comprising self-attention and feed-forward sublayers with optional caching.

    Attributes:
        attention (ScaledDotProdAttV2): Self-attention module handling rotary embeddings and caches.
        feed_forward (nn.Module): Feed-forward stack applied after attention.
        att_norm (nn.Module): Normalization layer applied before self-attention.
        ff_norm (nn.Module): Normalization layer applied before the feed-forward stack.
        drop_path (Optional[DropPath1d]): Stochastic depth module applied to the residual output.
    """

    def __init__(
        self,
        att_type: TransformerV2AttType,
        ff_type: TransformerV2FeedForwardType,
        num_feats: int,
        num_heads: int,
        num_kv_heads: int,
        ff_intermediate_feats: int,
        ff_kernel_size: int,
        ff_dilation: int,
        ff_activation: Union[str, nn.Module] = "silu",
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        rope: Optional[RotaryPosEncoder] = None,
        is_causal: bool = False,
        att_sliding_window: Optional[int] = None,
        sdp_backend: SDPBackendType = SDPBackendType.default(),
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-5,
        model_parallel: bool = False,
    ):
        """Configure the self-attention transformer block.

        Args:
            att_type (TransformerV2AttType): Attention implementation to instantiate.
            ff_type (TransformerV2FeedForwardType): Feed-forward module implementation to instantiate.
            num_feats (int): Hidden size of the block inputs.
            num_heads (int): Number of attention heads for the query stream.
            num_kv_heads (int): Number of key/value heads (may differ when using grouped attention).
            ff_intermediate_feats (int): Feed-forward network width before projection.
            ff_kernel_size (int): Kernel size for convolutional feed-forward variants.
            ff_dilation (int): Dilation factor for convolutional feed-forward variants.
            ff_activation (Union[str, nn.Module], optional): Feed-forward activation identifier. Defaults to ``"silu"``.
            ff_bias (bool, optional): Whether feed-forward linear layers include bias terms. Defaults to ``False``.
            ff_multiple_of (int, optional): Rounds ``ff_intermediate_feats`` up to the nearest multiple. Defaults to ``256``.
            att_dropout_rate (float, optional): Dropout probability applied to attention weights. Defaults to ``0.0``.
            att_bias (bool, optional): Whether attention projection layers include biases. Defaults to ``False``.
            rope (Optional[RotaryPosEncoder], optional): Rotary position encoder applied to attention logits.
            is_causal (bool, optional): If ``True``, enables causal masking within the attention module. Defaults to ``False``.
            att_sliding_window (Optional[int], optional): Optional sliding-window constraint for attention. Defaults to ``None``.
            sdp_backend (SDPBackendType, optional): Preferred scaled dot-product backend. Defaults to ``SDPBackendType.default()``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
            drop_path_rate (float, optional): Stochastic depth rate applied to the residual branch. Defaults to ``0.0``.
            norm_eps (float, optional): Epsilon for the normalization layers. Defaults to ``1e-5``.
            model_parallel (bool, optional): Whether to use tensor model-parallel attention projections. Defaults to ``False``.
        """
        super().__init__()
        att_class = TransformerV2AttType.to_class(att_type)
        ff_class = TransformerV2FeedForwardType.to_class(ff_type)
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.att_norm = norm_layer(num_feats, norm_eps)
        self.ff_norm = norm_layer(num_feats, norm_eps)

        self.attention = att_class(
            num_feats=num_feats,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=att_dropout_rate,
            att_bias=att_bias,
            rope=rope,
            is_causal=is_causal,
            sliding_window=att_sliding_window,
            sdp_backend=sdp_backend,
            model_parallel=model_parallel,
        )
        self.feed_forward = ff_class(
            num_feats,
            ff_intermediate_feats,
            activation=ff_activation,
            kernel_size=ff_kernel_size,
            dilation=ff_dilation,
            ff_bias=ff_bias,
            ff_multiple_of=ff_multiple_of,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath1d(drop_path_rate) if drop_path_rate > 0.0 else None

    def init_state(
        self,
        batch_size: int,
        max_cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """Initialize the key/value cache dictionary required for streaming attention.

        Args:
            batch_size (int): Maximum batch size the cache must support.
            max_cache_length (int): Maximum number of cached timesteps.
            device (Optional[torch.device]): Device where the cache is allocated. Defaults to the attention weight device.
            dtype (Optional[torch.dtype]): Tensor dtype to use for the cache. Defaults to the attention weight dtype.

        Returns:
            Dict[str, torch.Tensor]: Cache dictionary with keys ``"k"``, ``"v"``, and ``"cache_length"``.
        """

        if not hasattr(self.attention, "init_state"):
            raise AttributeError(
                "Attention module does not support cache initialization."
            )
        return self.attention.init_state(
            batch_size=batch_size,
            max_cache_length=max_cache_length,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Run self-attention and feed-forward sublayers with residual connections.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, hidden_dim)`.
            x_mask (Optional[torch.Tensor]): Optional attention mask broadcastable to the attention scores.
            start_pos (int, optional): Starting position for rotary embeddings / cache writes. Defaults to 0.
            state (Optional[Dict[str, torch.Tensor]]): Optional cache dictionary produced by :meth:`init_state`.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]: Output tensor and, if ``state`` is
            provided, the updated cache dictionary.
        """
        x_norm = self.att_norm(x)
        att_out = self.attention(
            x_norm,
            x_norm,
            x_norm,
            x_mask,
            start_pos,
            start_pos,
            state=state,
        )
        if isinstance(att_out, tuple):
            att_value, new_state = att_out
        else:
            att_value = att_out
            new_state = None
        h = x + att_value
        out = h + self.feed_forward(self.ff_norm(h))
        if self.drop_path is not None and self.training:
            out = x + self.drop_path(out - x)
        if new_state is not None:
            return out, new_state
        return out


class TransformerV2CrossAttBlock(nn.Module):
    """Transformer block combining self-attention, cross-attention, and feed-forward sublayers with cache support.

    Attributes:
        attention (ScaledDotProdAttV2): Self-attention module operating on the query stream.
        cross_attention (ScaledDotProdAttV2): Cross-attention module operating on key/value streams.
        att_norm (nn.Module): Normalization layer applied before self-attention.
        cross_att_q_norm (nn.Module): Normalization applied to queries before cross-attention.
        cross_att_kv_norm (nn.Module): Normalization applied to keys/values before cross-attention.
        feed_forward (nn.Module): Feed-forward stack applied after attention.
        drop_path (Optional[DropPath1d]): Stochastic depth module applied to the residual output.
    """

    def __init__(
        self,
        att_type: TransformerV2AttType,
        ff_type: TransformerV2FeedForwardType,
        num_feats: int,
        num_heads: int,
        num_kv_feats: int,
        num_kv_heads: int,
        ff_intermediate_feats: int,
        ff_kernel_size: int,
        ff_dilation: int,
        ff_activation: Union[str, nn.Module] = "silu",
        ff_bias: bool = False,
        ff_multiple_of: int = 256,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        rope: Optional[RotaryPosEncoder] = None,
        rope_in_self_att: bool = True,
        rope_in_cross_att: bool = True,
        norm_layer: Optional[Type[nn.Module]] = None,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-5,
        model_parallel: bool = False,
    ):
        """Configure the cross-attention transformer block.

        Args:
            att_type (TransformerV2AttType): Attention implementation to instantiate for both paths.
            ff_type (TransformerV2FeedForwardType): Feed-forward module implementation to instantiate.
            num_feats (int): Hidden size of the self-attention stream.
            num_heads (int): Number of attention heads for the query stream.
            num_kv_feats (int): Feature dimension of the cross-attention key/value stream.
            num_kv_heads (int): Number of key/value heads (may differ when using grouped attention).
            ff_intermediate_feats (int): Feed-forward network width before projection.
            ff_kernel_size (int): Kernel size for convolutional feed-forward variants.
            ff_dilation (int): Dilation factor for convolutional feed-forward variants.
            ff_activation (Union[str, nn.Module], optional): Feed-forward activation identifier. Defaults to ``"silu"``.
            ff_bias (bool, optional): Whether feed-forward linear layers include bias terms. Defaults to ``False``.
            ff_multiple_of (int, optional): Rounds ``ff_intermediate_feats`` up to the nearest multiple. Defaults to ``256``.
            att_dropout_rate (float, optional): Dropout probability applied to attention weights. Defaults to ``0.0``.
            att_bias (bool, optional): Whether attention projection layers include biases. Defaults to ``False``.
            rope (Optional[RotaryPosEncoder], optional): Shared rotary position encoder instance.
            rope_in_self_att (bool, optional): If ``True``, applies the shared RoPE to self-attention. Defaults to ``True``.
            rope_in_cross_att (bool, optional): If ``True``, applies the shared RoPE to cross-attention. Defaults to ``True``.
            norm_layer (Optional[Type[nn.Module]], optional): Normalization constructor; :class:`nn.LayerNorm` if ``None``.
            drop_path_rate (float, optional): Stochastic depth rate applied to the residual branch. Defaults to ``0.0``.
            norm_eps (float, optional): Epsilon for the normalization layers. Defaults to ``1e-5``.
            model_parallel (bool, optional): Whether to use tensor model-parallel attention projections. Defaults to ``False``.
        """
        super().__init__()
        att_class = TransformerV2AttType.to_class(att_type)
        ff_class = TransformerV2FeedForwardType.to_class(ff_type)
        if norm_layer is None:
            norm_layer = nn.LayerNorm

        self.att_norm = norm_layer(num_feats, norm_eps)
        self.cross_att_q_norm = norm_layer(num_feats, norm_eps)
        self.cross_att_kv_norm = norm_layer(num_kv_feats, norm_eps)
        self.ff_norm = norm_layer(num_feats, norm_eps)

        self.attention = att_class(
            num_feats=num_feats,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=att_dropout_rate,
            att_bias=att_bias,
            rope=rope if rope_in_self_att else None,
            model_parallel=model_parallel,
        )

        self.cross_attention = att_class(
            num_feats=num_feats,
            num_heads=num_heads,
            num_kv_feats=num_kv_feats,
            num_kv_heads=num_kv_heads,
            dropout_rate=att_dropout_rate,
            att_bias=att_bias,
            rope=rope if rope_in_cross_att else None,
            model_parallel=model_parallel,
        )

        self.feed_forward = ff_class(
            num_feats,
            ff_intermediate_feats,
            activation=ff_activation,
            kernel_size=ff_kernel_size,
            dilation=ff_dilation,
            ff_bias=ff_bias,
            ff_multiple_of=ff_multiple_of,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath1d(drop_path_rate) if drop_path_rate > 0.0 else None

    def init_state(
        self,
        batch_size: int,
        self_max_cache_length: int,
        cross_max_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Initialize caches for both self- and cross-attention paths.

        Returns a mapping with keys `self_att` and/or `cross_att` when the respective
        attention modules implement cache initialization.

        Args:
            batch_size (int): Maximum batch size the caches must support.
            self_max_cache_length (int): Maximum cache length for the self-attention stream.
            cross_max_cache_length (Optional[int]): Maximum cache length for the cross-attention stream.
                Defaults to ``self_max_cache_length`` when ``None``.
            device (Optional[torch.device]): Device where caches are allocated.
            dtype (Optional[torch.dtype]): Tensor dtype for caches.

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: Dictionary containing cache dictionaries for self-attention
            (key ``"self_att"``) and cross-attention (key ``"cross_att"``) when supported.
        """

        state: Dict[str, Dict[str, torch.Tensor]] = {}
        if cross_max_cache_length is None:
            cross_max_cache_length = self_max_cache_length

        if hasattr(self.attention, "init_state"):
            state["self_att"] = self.attention.init_state(
                batch_size=batch_size,
                max_cache_length=self_max_cache_length,
                device=device,
                dtype=dtype,
            )
        if hasattr(self.cross_attention, "init_state"):
            state["cross_att"] = self.cross_attention.init_state(
                batch_size=batch_size,
                max_cache_length=cross_max_cache_length,
                device=device,
                dtype=dtype,
            )
        return state

    def forward(
        self,
        x: torch.Tensor,
        x_mask: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
        x_kv_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
        start_pos_kv: int = 0,
        state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        """Run self- then cross-attention (if provided) followed by the feed-forward stack.

        Args:
            x (torch.Tensor): Query tensor of shape `(batch, seq_len, hidden_dim)`.
            x_mask (Optional[torch.Tensor]): Optional mask applied during self-attention.
            x_kv (Optional[torch.Tensor]): Key/value tensor for cross-attention. If ``None`` the cross step is skipped.
            x_kv_mask (Optional[torch.Tensor]): Optional mask applied during cross-attention.
            start_pos (int, optional): Starting position for self-attention cache writes. Defaults to 0.
            start_pos_kv (int, optional): Starting position for cross-attention cache writes. Defaults to 0.
            state (Optional[Dict[str, Dict[str, torch.Tensor]]]): Optional cache dictionary returned by
                :meth:`init_state`.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]: Output tensor and, if a
            cache ``state`` is provided, the updated cache dictionary containing ``"self_att"`` and/or ``"cross_att"``.
        """
        x_norm = self.att_norm(x)
        self_state = state["self_att"] if state and "self_att" in state else None
        att_out = self.attention(
            x_norm,
            x_norm,
            x_norm,
            x_mask,
            start_pos,
            start_pos,
            state=self_state,
        )
        new_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
        if isinstance(att_out, tuple):
            att_value, updated_self_state = att_out
            new_state = {"self_att": updated_self_state}
        else:
            att_value = att_out
        h = x + att_value
        if x_kv is not None:
            h_norm = self.cross_att_q_norm(h)
            x_kv_norm = self.cross_att_kv_norm(x_kv)
            cross_state = (
                state.get("cross_att") if state and "cross_att" in state else None
            )
            cross_out = self.cross_attention(
                h_norm,
                x_kv_norm,
                x_kv_norm,
                x_kv_mask,
                query_start_pos=start_pos,
                key_start_pos=start_pos_kv,
                state=cross_state,
            )
            if isinstance(cross_out, tuple):
                cross_value, updated_cross_state = cross_out
                if new_state is None:
                    new_state = {}
                new_state["cross_att"] = updated_cross_state
            else:
                cross_value = cross_out
            h = h + cross_value

        out = h + self.feed_forward(self.ff_norm(h))
        if self.drop_path is not None and self.training:
            out = x + self.drop_path(out - x)
        if new_state is not None:
            return out, new_state
        return out


# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim=None,
#         max_position_embeddings=2048,
#         base=10000,
#         device=None,
#         scaling_factor=1.0,
#         rope_type="default",
#         config: Optional[LlamaConfig] = None,
#     ):
#         super().__init__()
#         # TODO (joao): remove the `if` below, only used for BC
#         self.rope_kwargs = {}
#         if config is None:
#             logger.warning_once(
#                 "`LlamaRotaryEmbedding` can now be fully parameterized by passing the model config through the "
#                 "`config` argument. All other arguments will be removed in v4.45"
#             )
#             self.rope_kwargs = {
#                 "rope_type": rope_type,
#                 "factor": scaling_factor,
#                 "dim": dim,
#                 "base": base,
#                 "max_position_embeddings": max_position_embeddings,
#             }
#             self.rope_type = rope_type
#             self.max_seq_len_cached = max_position_embeddings
#             self.original_max_seq_len = max_position_embeddings
#         else:
#             # BC: "rope_type" was originally "type"
#             if config.rope_scaling is not None:
#                 self.rope_type = config.rope_scaling.get(
#                     "rope_type", config.rope_scaling.get("type")
#                 )
#             else:
#                 self.rope_type = "default"
#             self.max_seq_len_cached = config.max_position_embeddings
#             self.original_max_seq_len = config.max_position_embeddings

#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

#         inv_freq, self.attention_scaling = self.rope_init_fn(
#             self.config, device, **self.rope_kwargs
#         )
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq

#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(
#                 self.config, device, seq_len=seq_len, **self.rope_kwargs
#             )
#             self.register_buffer(
#                 "inv_freq", inv_freq, persistent=False
#             )  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len

#         if (
#             seq_len < self.original_max_seq_len
#             and self.max_seq_len_cached > self.original_max_seq_len
#         ):  # reset
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len

#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_ids, device=x.device)

#         # Core RoPE block
#         inv_freq_expanded = (
#             self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         )
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = (
#             device_type
#             if isinstance(device_type, str) and device_type != "mps"
#             else "cpu"
#         )
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (
#                 inv_freq_expanded.float() @ position_ids_expanded.float()
#             ).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()

#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling

#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
#     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
#     t = torch.arange(end, device=freqs.device, dtype=torch.float32)
#     freqs = torch.outer(t, freqs)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
#     return freqs_cis


# def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
#     ndim = x.ndim
#     assert 0 <= 1 < ndim
#     assert freqs_cis.shape == (x.shape[1], x.shape[-1])
#     shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
#     return freqs_cis.view(*shape)


# def apply_rotary_emb(
#     xq: torch.Tensor,
#     xk: torch.Tensor,
#     freqs_cis: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
#     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
#     freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
#     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
#     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
#     return xq_out.type_as(xq), xk_out.type_as(xk)


# # meta
# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
#     bs, slen, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     return (
#         x[:, :, :, None, :]
#         .expand(bs, slen, n_kv_heads, n_rep, head_dim)
#         .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
#     )

# # hf
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(
#         batch, num_key_value_heads, n_rep, slen, head_dim
#     )
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# class Attention(nn.Module):
#     def __init__(self, args: ModelArgs):
#         super().__init__()
#         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
#         model_parallel_size = fs_init.get_model_parallel_world_size()
#         self.n_local_heads = args.n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = args.dim // args.n_heads

#         self.wq = ColumnParallelLinear(
#             args.dim,
#             args.n_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wk = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wv = ColumnParallelLinear(
#             args.dim,
#             self.n_kv_heads * self.head_dim,
#             bias=False,
#             gather_output=False,
#             init_method=lambda x: x,
#         )
#         self.wo = RowParallelLinear(
#             args.n_heads * self.head_dim,
#             args.dim,
#             bias=False,
#             input_is_parallel=True,
#             init_method=lambda x: x,
#         )

#         self.cache_k = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()
#         self.cache_v = torch.zeros(
#             (
#                 args.max_batch_size,
#                 args.max_seq_len,
#                 self.n_local_kv_heads,
#                 self.head_dim,
#             )
#         ).cuda()

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         bsz, seqlen, _ = x.shape
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

#         self.cache_k = self.cache_k.to(xq)
#         self.cache_v = self.cache_v.to(xq)

#         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
#         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

#         keys = self.cache_k[:bsz, : start_pos + seqlen]
#         values = self.cache_v[:bsz, : start_pos + seqlen]

#         # repeat k/v heads if n_kv_heads < n_heads
#         keys = repeat_kv(
#             keys, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
#         values = repeat_kv(
#             values, self.n_rep
#         )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         values = values.transpose(
#             1, 2
#         )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
#         scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
#         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#         output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
#         return self.wo(output)


# class LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         if layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True

#         self.q_proj = nn.Linear(
#             self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
#         )
#         self.k_proj = nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.v_proj = nn.Linear(
#             self.hidden_size,
#             self.num_key_value_heads * self.head_dim,
#             bias=config.attention_bias,
#         )
#         self.o_proj = nn.Linear(
#             self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
#         )

#         # TODO (joao): remove in v4.45 (RoPE is computed in the model, not in the decoder layers)
#         self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         if self.config.pretraining_tp > 1:
#             key_value_slicing = (
#                 self.num_key_value_heads * self.head_dim
#             ) // self.config.pretraining_tp
#             query_slices = self.q_proj.weight.split(
#                 (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#             )
#             key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#             value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

#             query_states = [
#                 F.linear(hidden_states, query_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             query_states = torch.cat(query_states, dim=-1)

#             key_states = [
#                 F.linear(hidden_states, key_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             key_states = torch.cat(key_states, dim=-1)

#             value_states = [
#                 F.linear(hidden_states, value_slices[i])
#                 for i in range(self.config.pretraining_tp)
#             ]
#             value_states = torch.cat(value_states, dim=-1)

#         else:
#             query_states = self.q_proj(hidden_states)
#             key_states = self.k_proj(hidden_states)
#             value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)
#         attn_weights = torch.matmul(
#             query_states, key_states.transpose(2, 3)
#         ) / math.sqrt(self.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(
#             attn_weights, dim=-1, dtype=torch.float32
#         ).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(
#             attn_weights, p=self.attention_dropout, training=self.training
#         )
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, -1)

#         if self.config.pretraining_tp > 1:
#             attn_output = attn_output.split(
#                 self.hidden_size // self.config.pretraining_tp, dim=2
#             )
#             o_proj_slices = self.o_proj.weight.split(
#                 self.hidden_size // self.config.pretraining_tp, dim=1
#             )
#             attn_output = sum(
#                 [
#                     F.linear(attn_output[i], o_proj_slices[i])
#                     for i in range(self.config.pretraining_tp)
#                 ]
#             )
#         else:
#             attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class LlamaFlashAttention2(LlamaAttention):
#     """
#     Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
#     untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
#     flash attention and deal with padding tokens in case the input contains any of them.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
#         # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
#         # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
#         self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if isinstance(past_key_value, StaticCache):
#             raise ValueError(
#                 "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
#                 "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
#             )

#         output_attentions = False

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         # Flash attention requires the input to have the shape
#         # batch_size x seq_length x head_dim x hidden_dim
#         # therefore we just need to keep the original shape
#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
#         # to be able to avoid many of these transpose/reshape/view.
#         query_states = query_states.transpose(1, 2)
#         key_states = key_states.transpose(1, 2)
#         value_states = value_states.transpose(1, 2)

#         dropout_rate = self.attention_dropout if self.training else 0.0

#         # In PEFT, usually we cast the layer norms in float32 for training stability reasons
#         # therefore the input hidden states gets silently casted in float32. Hence, we need
#         # cast them back in the correct dtype just to be sure everything works as expected.
#         # This might slowdown training & inference so it is recommended to not cast the LayerNorms
#         # in fp32. (LlamaRMSNorm handles it correctly)

#         input_dtype = query_states.dtype
#         if input_dtype == torch.float32:
#             if torch.is_autocast_enabled():
#                 target_dtype = torch.get_autocast_gpu_dtype()
#             # Handle the case where the model is quantized
#             elif hasattr(self.config, "_pre_quantization_dtype"):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype

#             logger.warning_once(
#                 f"The input hidden states seems to be silently casted in float32, this might be related to"
#                 f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
#                 f" {target_dtype}."
#             )

#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)

#         attn_output = _flash_attention_forward(
#             query_states,
#             key_states,
#             value_states,
#             attention_mask,
#             q_len,
#             position_ids=position_ids,
#             dropout=dropout_rate,
#             sliding_window=getattr(self, "sliding_window", None),
#             use_top_left_mask=self._flash_attn_uses_top_left_mask,
#             is_causal=self.is_causal,
#         )

#         attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
#         attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value


# class LlamaSdpaAttention(LlamaAttention):
#     """
#     Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
#     `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
#     SDPA API.
#     """

#     # Adapted from LlamaAttention.forward
#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         if output_attentions:
#             # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
#             logger.warning_once(
#                 "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
#                 'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return super().forward(
#                 hidden_states=hidden_states,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_value=past_key_value,
#                 output_attentions=output_attentions,
#                 use_cache=use_cache,
#                 cache_position=cache_position,
#                 position_embeddings=position_embeddings,
#             )

#         bsz, q_len, _ = hidden_states.size()

#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(
#             bsz, q_len, self.num_heads, self.head_dim
#         ).transpose(1, 2)
#         key_states = key_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)
#         value_states = value_states.view(
#             bsz, q_len, self.num_key_value_heads, self.head_dim
#         ).transpose(1, 2)

#         if position_embeddings is None:
#             logger.warning_once(
#                 "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
#                 "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
#                 "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
#                 "removed and `position_embeddings` will be mandatory."
#             )
#             cos, sin = self.rotary_emb(value_states, position_ids)
#         else:
#             cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(
#             query_states, key_states, cos, sin
#         )

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(
#                 key_states, value_states, self.layer_idx, cache_kwargs
#             )

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         causal_mask = attention_mask
#         if attention_mask is not None:
#             causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

#         # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
#         # Reference: https://github.com/pytorch/pytorch/issues/112577.
#         if query_states.device.type == "cuda" and causal_mask is not None:
#             query_states = query_states.contiguous()
#             key_states = key_states.contiguous()
#             value_states = value_states.contiguous()

#         # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
#         # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
#         is_causal = True if causal_mask is None and q_len > 1 else False

#         attn_output = torch.nn.functional.scaled_dot_product_attention(
#             query_states,
#             key_states,
#             value_states,
#             attn_mask=causal_mask,
#             dropout_p=self.attention_dropout if self.training else 0.0,
#             is_causal=is_causal,
#         )

#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.view(bsz, q_len, -1)

#         attn_output = self.o_proj(attn_output)

#         return attn_output, None, past_key_value


# LLAMA_ATTENTION_CLASSES = {
#     "eager": LlamaAttention,
#     "flash_attention_2": LlamaFlashAttention2,
#     "sdpa": LlamaSdpaAttention,
# }


# class TransformerBlock(nn.Module):
#     def __init__(self, layer_id: int, args: ModelArgs):
#         super().__init__()
#         self.n_heads = args.n_heads
#         self.dim = args.dim
#         self.head_dim = args.dim // args.n_heads
#         self.attention = Attention(args)
#         self.feed_forward = FeedForward(
#             dim=args.dim,
#             hidden_dim=4 * args.dim,
#             multiple_of=args.multiple_of,
#             ffn_dim_multiplier=args.ffn_dim_multiplier,
#         )
#         self.layer_id = layer_id
#         self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
#         self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

#     def forward(
#         self,
#         x: torch.Tensor,
#         start_pos: int,
#         freqs_cis: torch.Tensor,
#         mask: Optional[torch.Tensor],
#     ):
#         h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
#         out = h + self.feed_forward(self.ffn_norm(h))
#         return out


# class LlamaDecoderLayer(nn.Module):
#     def __init__(self, config: LlamaConfig, layer_idx: int):
#         super().__init__()
#         self.hidden_size = config.hidden_size

#         self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](
#             config=config, layer_idx=layer_idx
#         )

#         self.mlp = LlamaMLP(config)
#         self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
#         self.post_attention_layernorm = LlamaRMSNorm(
#             config.hidden_size, eps=config.rms_norm_eps
#         )

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: Optional[bool] = False,
#         use_cache: Optional[bool] = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         position_embeddings: Optional[
#             Tuple[torch.Tensor, torch.Tensor]
#         ] = None,  # will become mandatory in v4.45
#         **kwargs,
#     ) -> Tuple[
#         torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
#     ]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`, *optional*):
#                 attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
#                 query_sequence_length, key_sequence_length)` if default attention is used.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#             use_cache (`bool`, *optional*):
#                 If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
#                 (see `past_key_values`).
#             past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
#             cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
#                 Indices depicting the position of the input sequence tokens in the sequence
#             position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
#                 Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
#                 with `head_dim` being the embedding dimension of each attention head.
#             kwargs (`dict`, *optional*):
#                 Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
#                 into the model
#         """
#         residual = hidden_states

#         hidden_states = self.input_layernorm(hidden_states)

#         # Self Attention
#         hidden_states, self_attn_weights, present_key_value = self.self_attn(
#             hidden_states=hidden_states,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_value=past_key_value,
#             output_attentions=output_attentions,
#             use_cache=use_cache,
#             cache_position=cache_position,
#             position_embeddings=position_embeddings,
#             **kwargs,
#         )
#         hidden_states = residual + hidden_states

#         # Fully Connected
#         residual = hidden_states
#         hidden_states = self.post_attention_layernorm(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (self_attn_weights,)

#         if use_cache:
#             outputs += (present_key_value,)

#         return outputs

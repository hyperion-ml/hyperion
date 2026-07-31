"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks import ConformerDecoderBlockV1 as DBlock
from ..layer_blocks import TransformerConv1dSubsampler as Conv1dSubsampler
from ..layer_blocks import TransformerConv2dSubsampler as Conv2dSubsampler
from ..layers import ActivationFactory as AF
from ..layers import ConvPosEncoder, NoPosEncoder
from ..layers import NormLayer1dFactory as NLF
from ..layers import PosEncoder, RelPosEncoder
from ..utils import make_attn_mask_causal, scale_seq_lengths, seq_lengths_to_mask
from .net_arch import NetArch


class ConformerDecoderV1(NetArch):
    """Conformer decoder mixing Transformer Decoder with Conformer Encoder Conv blocks

    This becomes a standard Transformer Decoder by setting conv_repeats=0, pos_enc_type='abs', ff_macaron=False.

    Attributes:
      in_feats: input features dimension
      d_model: encoder blocks feature dimension
      num_heads: number of heads
      num_blocks: number of self attn blocks
      self_att_type: string in ['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1', 'block-scaled-dot-prod-v1']
      self_att_context: maximum context range for local attention
      cross_att_type: string in ['scaled-dot-prod-v1', 'local-scaled-dot-prod-v1', 'block-scaled-dot-prod-v1']
      conv_repeats: number of conv blocks in each conformer block
      conv_kernel_sizes: kernel size for conv blocks
      conv_strides: stride for depth-wise conv in the first conv block of each conformer block
      ff_type: string in ['linear', 'conv1dx2', 'conv1d-linear']
      d_ff: dimension of middle layer in feed_forward block
      ff_kernel_size: kernel size for convolutional versions of ff block
      dropout_rate: dropout rate for ff and conv blocks
      pos_dropout_rate: dropout rate for positional encoder
      att_dropout_rate: dropout rate for attention block
      in_layer_type: input layer block type in ['linear','conv2d-sub', 'embed', None]
      pos_enc_type: type of positional encoder ['no', 'abs', 'rel', 'conv']

      causal_pos_enc: if True, use causal positional encodings (when rel_pos_enc=True), it assumes
                      that query q_i only attents to key k_j when j<=i
      hid_act:  hidden activations in ff and input blocks
      conv_norm_layer: norm layer constructor or str for conv block,
                       if None it uses BatchNorm1d
      se_r:         Squeeze-Excitation compression ratio,
                    if None it doesn't use Squeeze-Excitation
      ff_macaron: if True, it uses macaron-net style ff layers, otherwise transformer style.
      red_lnorms:  it True, use redundant LNorm layers at the output of the conformer blocks as
                  in the paper
      concat_after: if True, if concats attention input and output and apply linear transform, i.e.,
                             y = x + linear(concat(x, att(x)))
                    if False, y = x + att(x)
      padding_idx: padding idx for embed layer
      in_time_dim: time dimension in the input Tensor
      src_time_dim: time dimension in the source Tensor
      out_time_dim: dimension that we want to be time in the output tensor
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_blocks: int = 6,
        self_att_type: str = "scaled-dot-prod-v1",
        att_context: int = 25,
        cross_att_type: str = "scaled-dot-prod-v1",
        conv_repeats: Union[int, List[int]] = 0,
        conv_kernel_sizes: Union[int, List[int]] = 31,
        conv_strides: Union[int, List[int]] = 1,
        ff_type: str = "linear",
        d_ff: int = 2048,
        ff_kernel_size: int = 1,
        dropout_rate: float = 0.1,
        pos_dropout_rate: float = 0.1,
        att_dropout_rate: float = 0.0,
        in_layer_type: Union[str, nn.Module, None] = "embed",
        in_stride: int = 4,
        pos_enc_type: str = "abs",
        causal_pos_enc: bool = False,
        pos_kernel_size: int = 128,
        pos_num_groups: int = 16,
        hid_act: Any = "swish",
        conv_norm_layer: Optional[Any] = None,
        se_r: Optional[int] = None,
        ff_macaron: bool = True,
        red_lnorms: bool = True,
        concat_after: bool = False,
        padding_idx: int = -1,
        in_time_dim: int = 1,
        src_time_dim: int = 1,
        out_time_dim: int = 1,
        in_feats: Optional[int] = None,
        with_output: bool = True,
    ) -> None:
        """Initialize the conformer decoder architecture.

        Args:
          num_classes: Number of output classes.
          d_model: Hidden feature dimension used throughout the network.
          num_heads: Number of attention heads.
          num_blocks: Number of conformer decoder blocks.
          self_att_type: Self-attention implementation identifier.
          att_context: Local-attention context size.
          cross_att_type: Cross-attention implementation identifier.
          conv_repeats: Number of convolution sub-blocks per decoder block.
          conv_kernel_sizes: Convolution kernel size(s) per decoder block.
          conv_strides: Convolution stride(s) per decoder block.
          ff_type: Feed-forward block type.
          d_ff: Hidden dimension of the feed-forward block.
          ff_kernel_size: Kernel size for convolutional feed-forward variants.
          dropout_rate: Dropout probability for the blocks.
          pos_dropout_rate: Positional encoder dropout probability.
          att_dropout_rate: Attention dropout probability.
          in_layer_type: Input layer type.
          in_stride: Stride for subsampling input layers.
          pos_enc_type: Positional encoder type.
          causal_pos_enc: Whether to use causal relative positional encoding.
          pos_kernel_size: Kernel size for convolutional positional encoding.
          pos_num_groups: Number of groups for convolutional positional encoding.
          hid_act: Hidden activation specification.
          conv_norm_layer: Convolution normalization layer constructor or name.
          se_r: Optional squeeze-excitation reduction ratio.
          ff_macaron: Whether to use macaron feed-forward layers.
          red_lnorms: Whether to keep redundant layer norms on block outputs.
          concat_after: Whether to use concat-after attention.
          padding_idx: Padding index for the embedding input layer.
          in_time_dim: Time dimension index in the decoder input tensor.
          src_time_dim: Time dimension index in the source tensor.
          out_time_dim: Time dimension index in the output tensor.
          in_feats: Input feature dimension. Defaults to ``num_classes``.
          with_output: Whether to append the final linear output layer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.with_output = with_output
        if in_feats is None:
            in_feats = num_classes
        self.in_feats = in_feats
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.self_att_type = self_att_type
        self.cross_att_type = cross_att_type
        self.att_context = att_context

        self.conv_repeats = self._standarize_cblocks_param(
            conv_repeats, num_blocks, "conv_repeats"
        )
        self.conv_kernel_sizes = self._standarize_cblocks_param(
            conv_kernel_sizes, num_blocks, "conv_kernel_sizes"
        )
        self.conv_strides = self._standarize_cblocks_param(
            conv_strides, num_blocks, "conv_strides"
        )

        self.ff_type = ff_type
        self.d_ff = d_ff
        self.ff_kernel_size = ff_kernel_size
        self.dropout_rate = dropout_rate
        self.pos_enc_type = pos_enc_type
        self.causal_pos_enc = causal_pos_enc
        self.att_dropout_rate = att_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.in_layer_type = in_layer_type
        self.in_stride = in_stride
        self.se_r = se_r
        self.ff_macaron = ff_macaron
        self.red_lnorms = red_lnorms
        self.concat_after = concat_after
        self.padding_idx = padding_idx
        self.in_time_dim = in_time_dim
        self.src_time_dim = src_time_dim
        self.out_time_dim = out_time_dim
        self.hid_act = hid_act
        self.pos_kernel_size = pos_kernel_size
        self.pos_num_groups = pos_num_groups

        self.conv_norm_layer = conv_norm_layer
        norm_groups = None
        if conv_norm_layer == "group-norm":
            norm_groups = min(d_model // 2, 32)
        self._conv_norm_layer = NLF.create(conv_norm_layer, norm_groups)

        self._make_in_layer()

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                DBlock(
                    d_model,
                    self_att_type,
                    cross_att_type,
                    num_heads,
                    self.conv_repeats[i],
                    self.conv_kernel_sizes[i],
                    self.conv_strides[i],
                    ff_type,
                    d_ff,
                    ff_kernel_size,
                    hid_act=hid_act,
                    dropout_rate=dropout_rate,
                    att_context=att_context,
                    att_dropout_rate=att_dropout_rate,
                    pos_enc_type=pos_enc_type,
                    causal_pos_enc=causal_pos_enc,
                    conv_norm_layer=self._conv_norm_layer,
                    se_r=se_r,
                    ff_macaron=ff_macaron,
                    out_lnorm=self.red_lnorms,
                    concat_after=concat_after,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        if not self.red_lnorms:
            self.norm_out = nn.LayerNorm(d_model)

        if with_output:
            self.output_layer = nn.Linear(d_model, num_classes)

    @staticmethod
    def _standarize_cblocks_param(
        p: Union[int, List[int]], num_blocks: int, p_name: str
    ) -> List[int]:
        """Expand a per-block parameter to match the number of blocks.

        Args:
          p: Integer value or list of values to normalize.
          num_blocks: Number of decoder blocks.
          p_name: Parameter name used in error messages.

        Returns:
          List of length ``num_blocks``.
        """
        if isinstance(p, int):
            p = [p] * num_blocks
        elif isinstance(p, list):
            if len(p) == 1:
                p = p * num_blocks

            assert len(p) == num_blocks, "len(%s)(%d)!=%d" % (
                p_name,
                len(p),
                num_blocks,
            )
        else:
            raise TypeError("wrong type for param {}={}".format(p_name, p))

        return p

    def _make_in_layer(self) -> None:
        """Create the input projection or subsampling front-end."""
        in_feats = self.in_feats
        d_model = self.d_model
        dropout_rate = self.dropout_rate
        if self.pos_enc_type == "no":
            pos_enc = NoPosEncoder()
        elif self.pos_enc_type == "rel":
            pos_enc = RelPosEncoder(d_model, self.pos_dropout_rate)
        elif self.pos_enc_type == "abs":
            pos_enc = PosEncoder(d_model, self.pos_dropout_rate)
        elif self.pos_enc_type == "conv":
            pos_enc = ConvPosEncoder(
                d_model, self.pos_kernel_size, self.pos_num_groups, self.hid_act
            )
        else:
            raise Exception("wrong pos-enc-type={}".format(self.pos_enc_type))

        hid_act = AF.create(self.hid_act)

        if self.in_layer_type == "linear":
            self.in_layer = nn.Sequential(
                nn.Linear(in_feats, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout_rate),
                hid_act,
                pos_enc,
            )
        elif self.in_layer_type == "conv2d-sub":
            self.in_layer = Conv2dSubsampler(
                in_feats,
                d_model,
                hid_act,
                self.in_stride,
                pos_enc,
                time_dim=self.in_time_dim,
            )
        elif self.in_layer_type == "conv1d-sub":
            self.in_layer = Conv1dSubsampler(
                in_feats,
                d_model,
                hid_act,
                self.in_stride,
                pos_enc,
                time_dim=self.in_time_dim,
            )
        elif self.in_layer_type == "embed":
            self.in_layer = nn.Sequential(
                nn.Embedding(in_feats, d_model, padding_idx=self.padding_idx), pos_enc
            )
        elif isinstance(self.in_layer_type, nn.Module):
            self.in_layer = nn.Sequential(self.in_layer_type, pos_enc)
        elif self.in_layer_type is None:
            self.in_layer = pos_enc
        else:
            raise ValueError(f"unknown in_layer_type: {self.in_layer_type}")

    def _make_masks(
        self,
        batch_size: int,
        device: torch.device,
        max_in_length: int,
        x_lengths: Optional[torch.Tensor],
        x_mask: Optional[torch.Tensor],
        max_src_length: int,
        x_src_lengths: Optional[torch.Tensor],
        x_src_mask: Optional[torch.Tensor],
        causal_mask: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Create decoder and source masks from lengths when needed.

        Args:
          batch_size: Batch size of the decoder input.
          device: Device used to materialize masks when lengths are absent.
          max_in_length: Maximum input length.
          x_lengths: Input sequence lengths.
          x_mask: Optional precomputed input mask.
          max_src_length: Maximum source length.
          x_src_lengths: Source sequence lengths.
          x_src_mask: Optional precomputed source mask.
          causal_mask: Whether to apply a causal mask to the input side.

        Returns:
          A tuple ``(x_mask, x_src_mask)``.
        """
        if x_mask is None:
            if x_lengths is not None:
                x_mask = seq_lengths_to_mask(
                    x_lengths, max_in_length, time_dim=2, ndim=3
                )
            elif causal_mask:
                x_mask = torch.ones(
                    (batch_size, 1, max_in_length), dtype=torch.bool, device=device
                )

        if causal_mask and x_mask is not None:
            x_mask = make_attn_mask_causal(x_mask)

        if x_src_mask is None and x_src_lengths is not None:
            x_src_mask = seq_lengths_to_mask(
                x_src_lengths, max_src_length, time_dim=1, ndim=3
            )

        return x_mask, x_src_mask

    def _forward_input(
        self, x: torch.Tensor, x_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the decoder input front-end.

        Args:
          x: Input tensor.
          x_mask: Optional input mask.

        Returns:
          Tuple with the transformed tensor and updated mask.
        """
        if isinstance(self.in_layer, (Conv2dSubsampler, Conv1dSubsampler)):
            x, x_mask = self.in_layer(x, x_mask)
        else:
            if self.in_time_dim != 1:
                x = x.transpose(1, self.in_time_dim).contiguous()
            x = self.in_layer(x)

        return x, x_mask

    def forward(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        x_src_lengths: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        x_src_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run a forward pass through the decoder.

        Args:
          x: Input tensor with size ``(batch, time_out, num_feats)`` or
            ``(batch, time_out)``.
          x_src: Source tensor with size ``(batch, time_in, num_feats)``.
          x_lengths: Lengths of the input sequences.
          x_src_lengths: Lengths of the source sequences.
          x_mask: mask to indicate valid time steps for x (batch, time_out).
                  It overwrites the mask of x_lengths.
          x_src_mask: mask to indicate valid time steps for x_src (batch, time_in).
                  It overwrites the mask of x_src_lengths.
          causal_mask: Whether to apply a causal attention mask to ``x``.

        Returns:
           Tuple containing the output tensor and the scaled output lengths.
        """
        if self.src_time_dim != 1:
            x_src = x_src.transpose(1, self.src_time_dim).contiguous()

        max_in_length = x.size(self.in_time_dim)
        max_src_length = x_src.size(1)
        x_mask, x_src_mask = self._make_masks(
            x.size(0),
            x.device,
            max_in_length,
            x_lengths,
            x_mask,
            max_src_length,
            x_src_lengths,
            x_src_mask,
            causal_mask,
        )
        x, x_mask = self._forward_input(x, x_mask)

        if isinstance(x, tuple):
            x, pos_emb = x
            b_args = {"pos_emb": pos_emb}
        else:
            b_args = {}

        for i in range(len(self.blocks)):
            x, x_mask = self.blocks[i](
                x, x_src, mask=x_mask, mask_src=x_src_mask, **b_args
            )

        if not self.red_lnorms:
            x = self.norm_out(x)

        if self.with_output:
            x = self.output_layer(x)

        if x_lengths is not None:
            x_lengths = scale_seq_lengths(x_lengths, x.size(1), max_in_length)

        if self.out_time_dim != 1:
            x = x.transpose(1, self.out_time_dim)

        return x, x_lengths

    def forward_1step(
        self,
        x: torch.Tensor,
        x_src: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run a single-step forward pass for streaming inference.

        Args:
          x: Input tensor with size ``(batch, time, num_feats)``.
          x_src: Source tensor with size ``(batch, time_src, num_feats)``.
          x_lengths: Lengths of the input sequences.
          x_mask: Mask indicating valid time steps for ``x``.
                  It overwrites the mask of x_lengths.
          cache: Per-block streaming cache from the previous step.

        Returns:
           Tuple containing the last-step output tensor and the updated cache.
        """
        max_in_length = x.size(self.in_time_dim)
        if x_mask is None and x_lengths is not None:
            x_mask = seq_lengths_to_mask(x_lengths, max_in_length, time_dim=2, ndim=3)

        if self.src_time_dim != 1:
            x_src = x_src.transpose(1, self.src_time_dim).contiguous()

        x, x_mask = self._forward_input(x, x_mask)

        if isinstance(x, tuple):
            x, pos_emb = x
            b_args = {"pos_emb": pos_emb}
        else:
            b_args = {}

        if cache is None:
            cache = [None] * len(self.blocks)

        next_cache = []
        for i in range(len(self.blocks)):
            block_input = x
            x, x_mask = self.blocks[i](
                x, x_src, mask=x_mask, mask_src=None, cache=cache[i], **b_args
            )
            next_cache.append(block_input)

        if not self.red_lnorms:
            x = self.norm_out(x[:, -1])
        else:
            x = x[:, -1]

        if self.with_output:
            x = self.output_layer(x)

        return x, next_cache

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Get the network configuration.

        Args:
          no_class_name: Whether to omit the class name from the config.

        Returns:
          Dictionary with the decoder configuration.
        """
        config = {
            "num_classes": self.num_classes,
            "in_feats": self.in_feats,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "self_att_type": self.self_att_type,
            "cross_att_type": self.cross_att_type,
            "att_context": self.att_context,
            "conv_repeats": self.conv_repeats,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_strides": self.conv_strides,
            "ff_type": self.ff_type,
            "d_ff": self.d_ff,
            "ff_kernel_size": self.ff_kernel_size,
            "dropout_rate": self.dropout_rate,
            "att_dropout_rate": self.att_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "in_layer_type": self.in_layer_type,
            "in_stride": self.in_stride,
            "pos_enc_type": self.pos_enc_type,
            "causal_pos_enc": self.causal_pos_enc,
            "pos_kernel_size": self.pos_kernel_size,
            "pos_num_groups": self.pos_num_groups,
            "hid_act": self.hid_act,
            "se_r": self.se_r,
            "ff_macaron": self.ff_macaron,
            "red_lnorms": self.red_lnorms,
            "conv_norm_layer": self.conv_norm_layer,
            "concat_after": self.concat_after,
            "padding_idx": self.padding_idx,
            "in_time_dim": self.in_time_dim,
            "src_time_dim": self.src_time_dim,
            "out_time_dim": self.out_time_dim,
            "with_output": self.with_output,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def in_context(self) -> Tuple[int, int]:
        """Return the left and right input context required by the model."""
        return (self.att_context, self.att_context)

    def in_shape(self) -> Tuple[Optional[int], Optional[int], int]:
        """Input shape for network

        Returns:
           Tuple describing input shape
        """
        if self.in_time_dim == 1:
            return (None, None, self.in_feats)
        else:
            return (None, self.in_feats, None)

    def out_shape(
        self,
        in_shape: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None,
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Infers the network output shape given the input shape

        Args:
          in_shape: input shape tuple

        Returns:
          Tuple with the output shape
        """
        if in_shape is None:
            out_t = None
            batch_size = None
        else:
            assert len(in_shape) == 3
            batch_size = in_shape[0]
            in_t = in_shape[self.in_time_dim]
            if in_t is None:
                out_t = None
            else:
                if isinstance(self.in_layer, Conv2dSubsampler):
                    # out_t = in_t//4
                    out_t = ((in_t - 1) // 2 - 1) // 2
                else:
                    out_t = in_t

        if self.out_time_dim == 1:
            return (batch_size, out_t, self.d_model)
        else:
            return (batch_size, self.d_model, out_t)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments relevant to ``ConformerDecoderV1``.

        Args:
          kwargs: Argument dictionary.

        Returns:
          Filtered argument dictionary.
        """
        args = filter_func_args(ConformerDecoderV1.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: Any, prefix: Optional[str] = None, skip: Optional[Set[str]] = None
    ) -> None:
        """Add decoder configuration parameters to an argument parser.

        Args:
           parser: Argument parser instance.
           prefix: Prefix string to add to the argument names.
           skip: Set of argument names to omit.
        """
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats", type=int, default=None, help=("input feature dimension")
            )

        parser.add_argument(
            "--num-blocks", default=6, type=int, help=("number of tranformer blocks")
        )

        parser.add_argument(
            "--d-model", default=512, type=int, help=("encoder layer sizes")
        )

        parser.add_argument(
            "--num-heads",
            default=4,
            type=int,
            help=("number of heads in self-attention layers"),
        )

        parser.add_argument(
            "--self-att-type",
            default="scaled-dot-prod-v1",
            choices=[
                "scaled-dot-prod-v1",
                "local-scaled-dot-prod-v1",
                "block-scaled-dot-prod-v1",
            ],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--cross-att-type",
            default="scaled-dot-prod-v1",
            choices=[
                "scaled-dot-prod-v1",
                "local-scaled-dot-prod-v1",
                "block-scaled-dot-prod-v1",
            ],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--att-context",
            default=25,
            type=int,
            help=("context size when using local attention"),
        )

        parser.add_argument(
            "--conv-repeats",
            default=[0],
            type=int,
            nargs="+",
            help=("number of conv blocks in each conformer block"),
        )

        parser.add_argument(
            "--conv-kernel-sizes",
            default=[31],
            nargs="+",
            type=int,
            help=("kernels sizes for the depth-wise convs of each conformer block"),
        )

        parser.add_argument(
            "--conv-strides",
            default=[1],
            nargs="+",
            type=int,
            help=("resb-blocks strides for each encoder stage"),
        )

        parser.add_argument(
            "--ff-type",
            default="linear",
            choices=["linear", "conv1dx2", "conv1dlinear"],
            help=("type of feed forward layers in transformer block"),
        )

        parser.add_argument(
            "--d-ff",
            default=2048,
            type=int,
            help=("size middle layer in feed forward block"),
        )

        parser.add_argument(
            "--ff-kernel-size",
            default=3,
            type=int,
            help=("kernel size in convolutional feed forward block"),
        )

        parser.add_argument("--hid-act", default="swish", help="hidden activation")

        parser.add_argument(
            "--pos-dropout-rate",
            default=0.1,
            type=float,
            help="positional encoder dropout",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0, type=float, help="self-att dropout"
        )
        parser.add_argument(
            "--dropout-rate", default=0.1, type=float, help="feed-forward layer dropout"
        )

        parser.add_argument(
            "--in-layer-type",
            default="linear",
            choices=["embed", "linear", "conv2d-sub", "conv1d-sub"],
            help=("type of input layer"),
        )

        parser.add_argument(
            "--in-stride",
            default=4,
            type=int,
            choices=[1, 2, 4],
            help="stride of conformer input layer",
        )

        parser.add_argument(
            "--pos-enc-type",
            default="rel",
            choices=["no", "rel", "abs", "conv"],
            help=("type of positional encoder"),
        )

        parser.add_argument(
            "--causal-pos-enc",
            default=False,
            action=ActionYesNo,
            help="relative positional encodings are zero when attending to the future",
        )
        parser.add_argument(
            "--pos-kernel-size",
            default=128,
            type=int,
            help="kernel size for conv positional encoder",
        )
        parser.add_argument(
            "--pos-num-groups",
            default=16,
            type=int,
            help="number of conv groups for conv positional encoder",
        )

        parser.add_argument(
            "--conv-norm-layer",
            default=None,
            choices=[
                "batch-norm",
                "group-norm",
                "instance-norm",
                "instance-norm-affine",
                "layer-norm",
            ],
            help="type of normalization layer for conv block in conformer",
        )

        parser.add_argument(
            "--se-r",
            default=None,
            type=int,
            help=("squeeze-excitation compression ratio"),
        )

        parser.add_argument(
            "--ff-macaron",
            default=True,
            action=ActionYesNo,
            help="do not use macaron style ff layers ",
        )

        parser.add_argument(
            "--red-lnorms",
            default=True,
            action=ActionYesNo,
            help="use redundant Lnorm at conformer blocks' outputs",
        )

        parser.add_argument(
            "--concat-after",
            default=False,
            action=ActionYesNo,
            help="concatenate attention input and output instead of adding",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

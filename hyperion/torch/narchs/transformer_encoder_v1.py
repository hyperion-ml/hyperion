"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba, Nanxin Chen)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ..layer_blocks import TransformerConv2dSubsampler as Conv2dSubsampler
from ..layer_blocks import TransformerEncoderBlockV1 as EBlock
from ..layers import ActivationFactory as AF
from ..layers import PosEncoder, RelPosEncoder
from .net_arch import NetArch


class TransformerEncoderV1(NetArch):
    """Transformer encoder module.

    Attributes:
      in_feats: Input feature dimension.
      d_model: Encoder block feature dimension.
      num_heads: Number of attention heads.
      num_blocks: Number of self-attention blocks.
      att_type: Attention type in ["scaled-dot-prod-v1", "local-scaled-dot-prod-v1"].
      att_context: Maximum context range for local attention.
      ff_type: Feed-forward type in ["linear", "conv1dx2", "conv1dlinear", "conv1d-linear"].
      d_ff: Hidden dimension in the feed-forward block.
      ff_kernel_size: Kernel size for convolutional feed-forward variants.
      ff_dropout_rate: Dropout rate for feed-forward layers.
      pos_dropout_rate: Dropout rate for positional encoder.
      att_dropout_rate: Dropout rate for attention block.
      in_layer_type: Input layer type in ["linear", "conv2d-sub", "embed", None].
      rel_pos_enc: Whether to use relative positional encodings.
      causal_pos_enc: Whether relative positional encodings are causal.
      hid_act: Hidden activation used in feed-forward and input blocks.
      norm_before: Whether to apply layer normalization before sublayers.
      concat_after: Whether to concatenate attention input and output before projection.
      padding_idx: Padding index for the embedding input layer.
      in_time_dim: Time dimension in the input tensor.
      out_time_dim: Time dimension in the output tensor.
      in_layer: Input projection / subsampling module.
      blocks: Transformer encoder blocks stacked in the network.
      norm: Final layer normalization, created when ``norm_before`` is True.
    """

    def __init__(
        self,
        in_feats: int,
        d_model: int = 512,
        num_heads: int = 4,
        num_blocks: int = 6,
        att_type: str = "scaled-dot-prod-v1",
        att_context: int = 25,
        ff_type: str = "linear",
        d_ff: int = 2048,
        ff_kernel_size: int = 3,
        ff_dropout_rate: float = 0.1,
        pos_dropout_rate: float = 0.1,
        att_dropout_rate: float = 0.0,
        in_layer_type: Union[str, nn.Module, None] = "linear",
        rel_pos_enc: bool = False,
        causal_pos_enc: bool = False,
        hid_act: str = "relu",
        norm_before: bool = True,
        concat_after: bool = False,
        padding_idx: int = -1,
        in_time_dim: int = -1,
        out_time_dim: int = 1,
    ) -> None:
        """Initialize a transformer encoder architecture.

        Args:
          in_feats: Input feature dimension.
          d_model: Encoder block feature dimension.
          num_heads: Number of attention heads.
          num_blocks: Number of encoder blocks.
          att_type: Attention type string.
          att_context: Maximum context range for local attention.
          ff_type: Feed-forward type string.
          d_ff: Hidden dimension in the feed-forward block.
          ff_kernel_size: Kernel size for convolutional feed-forward variants.
          ff_dropout_rate: Dropout rate for feed-forward layers.
          pos_dropout_rate: Dropout rate for positional encoding layers.
          att_dropout_rate: Dropout rate for attention layers.
          in_layer_type: Input layer type or module.
          rel_pos_enc: Whether to use relative positional encodings.
          causal_pos_enc: Whether relative positional encodings are causal.
          hid_act: Hidden activation used in feed-forward and input blocks.
          norm_before: Whether to apply layer normalization before sublayers.
          concat_after: Whether to concatenate attention input and output before projection.
          padding_idx: Padding index for the embedding input layer.
          in_time_dim: Time dimension index in the input tensor.
          out_time_dim: Time dimension index in the output tensor.
        """

        super().__init__()
        self.in_feats = in_feats
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks

        self.att_type = att_type
        self.att_context = att_context

        if ff_type == "conv1dlinear":
            ff_type = "conv1d-linear"
        self.ff_type = ff_type
        self.d_ff = d_ff
        self.ff_kernel_size = ff_kernel_size
        self.ff_dropout_rate = ff_dropout_rate
        self.rel_pos_enc = rel_pos_enc
        self.causal_pos_enc = causal_pos_enc
        self.att_dropout_rate = att_dropout_rate
        self.pos_dropout_rate = pos_dropout_rate
        self.in_layer_type = in_layer_type
        self.norm_before = norm_before
        self.concat_after = concat_after
        self.padding_idx = padding_idx
        self.in_time_dim = in_time_dim
        self.out_time_dim = out_time_dim
        self.hid_act = hid_act

        self._make_in_layer()

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                EBlock(
                    d_model,
                    att_type,
                    num_heads,
                    ff_type,
                    d_ff,
                    ff_kernel_size,
                    ff_act=hid_act,
                    ff_dropout_rate=ff_dropout_rate,
                    att_context=att_context,
                    att_dropout_rate=att_dropout_rate,
                    rel_pos_enc=rel_pos_enc,
                    causal_pos_enc=causal_pos_enc,
                    norm_before=norm_before,
                    concat_after=concat_after,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        if self.norm_before:
            self.norm = nn.LayerNorm(d_model)

    def _make_in_layer(self) -> None:
        """Construct the input projection or subsampling layer."""

        in_feats = self.in_feats
        d_model = self.d_model
        dropout_rate = self.ff_dropout_rate
        if self.rel_pos_enc:
            pos_enc = RelPosEncoder(d_model, self.pos_dropout_rate)
        else:
            pos_enc = PosEncoder(d_model, self.pos_dropout_rate)

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
                in_feats, d_model, hid_act, pos_enc=pos_enc, time_dim=self.in_time_dim
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
            raise ValueError("unknown in_layer_type: " + self.in_layer_type)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        target_shape: Optional[Tuple[int, ...]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the transformer encoder forward pass.

        Args:
          x: Input tensor with shape ``(batch, time, num_feats)``.
          mask: Optional mask indicating valid time steps for ``x`` with shape
            ``(batch, time)``.
          target_shape: Reserved compatibility argument. It is currently ignored.

        Returns:
           Output tensor, or ``(output, mask)`` when a mask is provided.
        """
        if isinstance(self.in_layer, Conv2dSubsampler):
            x, mask = self.in_layer(x, mask)
        else:
            if self.in_time_dim != 1:
                x = x.transpose(1, self.in_time_dim).contiguous()
            x = self.in_layer(x)

        if isinstance(x, tuple):
            x, pos_emb = x
            b_args = {"pos_emb": pos_emb}
        else:
            b_args = {}

        for i in range(len(self.blocks)):
            x, mask = self.blocks[i](x, mask=mask, **b_args)

        if self.norm_before:
            x = self.norm(x)

        if self.out_time_dim != 1:
            x = x.transpose(1, self.out_time_dim)

        if mask is None:
            return x

        return x, mask

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Get the configuration dictionary for this architecture.

        Args:
          no_class_name: If ``True``, omit the class name entry from the base config.

        Returns:
          Dictionary with the serialized configuration parameters.
        """
        config = {
            "in_feats": self.in_feats,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "att_type": self.att_type,
            "att_context": self.att_context,
            "ff_type": self.ff_type,
            "d_ff": self.d_ff,
            "ff_kernel_size": self.ff_kernel_size,
            "ff_dropout_rate": self.ff_dropout_rate,
            "att_dropout_rate": self.att_dropout_rate,
            "pos_dropout_rate": self.pos_dropout_rate,
            "in_layer_type": self.in_layer_type,
            "rel_pos_enc": self.rel_pos_enc,
            "causal_pos_enc": self.causal_pos_enc,
            "hid_act": self.hid_act,
            "norm_before": self.norm_before,
            "concat_after": self.concat_after,
            "padding_idx": self.padding_idx,
            "in_time_dim": self.in_time_dim,
            "out_time_dim": self.out_time_dim,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def change_dropouts(
        self, pos_dropout_rate: float, att_dropout_rate: float, ff_dropout_rate: float
    ) -> None:
        """Update dropout rates in the encoder and its submodules.

        Args:
          pos_dropout_rate: New dropout rate for positional encoders.
          att_dropout_rate: New dropout rate for attention layers.
          ff_dropout_rate: New dropout rate for feed-forward layers.
        """

        assert pos_dropout_rate == 0 or self.pos_dropout_rate > 0
        assert att_dropout_rate == 0 or self.att_dropout_rate > 0
        assert ff_dropout_rate == 0 or self.ff_dropout_rate > 0

        for module in self.modules():
            if isinstance(module, PosEncoder):
                for layer in module.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = pos_dropout_rate

            elif isinstance(module, EBlock):
                for layer in module.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = ff_dropout_rate

                for layer in module.self_attn.modules():
                    if isinstance(layer, nn.Dropout):
                        layer.p = att_dropout_rate

        self.pos_dropout_rate = pos_dropout_rate
        self.att_dropout_rate = att_dropout_rate
        self.ff_dropout_rate = ff_dropout_rate

    def in_context(self) -> Tuple[int, int]:
        """Return the left/right temporal context required by the encoder.

        Returns:
          Tuple with the left and right context in frames.
        """
        return (self.att_context, self.att_context)

    def in_shape(self) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Return the expected input shape for the encoder.

        Returns:
           Tuple describing the input shape.
        """
        if self.in_time_dim == 1:
            return (None, None, self.in_feats)
        else:
            return (None, self.in_feats, None)

    def out_shape(
        self, in_shape: Optional[Tuple[Optional[int], Optional[int], Optional[int]]] = None
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Infer the output shape given an input shape.

        Args:
          in_shape: Input shape tuple.

        Returns:
          Tuple with the output shape.
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
        """Filter keyword arguments for :class:`TransformerEncoderV1`.

        Args:
          kwargs: Input argument dictionary.

        Returns:
          Dictionary containing only supported constructor arguments.
        """

        valid_args = (
            "num_blocks",
            "in_feats",
            "d_model",
            "num_heads",
            "att_type",
            "att_context",
            "ff_type",
            "d_ff",
            "ff_kernel_size",
            "ff_dropout_rate",
            "pos_dropout_rate",
            "att_dropout_rate",
            "in_layer_type",
            "hid_act",
            "rel_pos_enc",
            "causal_pos_enc",
            "concat_after",
            "padding_idx",
            "in_time_dim",
            "out_time_dim",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, in_feats: bool = False
    ) -> None:
        """Add transformer encoder config parameters to an argument parser.

        Args:
           parser: Argument parser to extend.
           prefix: Prefix string to add to the argument names.
           in_feats: Whether to expose the input feature dimension argument.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if in_feats:
            parser.add_argument(
                "--in-feats", type=int, default=80, help=("input feature dimension")
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
            "--att-type",
            default="scaled-dot-prod-v1",
            choices=["scaled-dot-prod-v1", "local-scaled-dot-prod-v1"],
            help=("type of self-attention"),
        )

        parser.add_argument(
            "--att-context",
            default=25,
            type=int,
            help=("context size when using local attention"),
        )

        parser.add_argument(
            "--ff-type",
            default="linear",
            choices=["linear", "conv1dx2", "conv1dlinear", "conv1d-linear"],
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

        try:
            parser.add_argument("--hid-act", default="relu", help="hidden activation")
        except:
            pass

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
            "--ff-dropout-rate",
            default=0.1,
            type=float,
            help="feed-forward layer dropout",
        )

        parser.add_argument(
            "--in-layer-type",
            default="linear",
            choices=["linear", "conv2d-sub"],
            help=("type of input layer"),
        )

        parser.add_argument(
            "--rel-pos-enc",
            default=False,
            action="store_true",
            help="use relative positional encoder",
        )

        parser.add_argument(
            "--causal-pos-enc",
            default=False,
            action="store_true",
            help="relative positional encodings are zero when attending to the future",
        )

        parser.add_argument(
            "--concat-after",
            default=False,
            action="store_true",
            help="concatenate attention input and output instead of adding",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='transformer encoder options')

    add_argparse_args = add_class_args

"""
Copyright 2024 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import List, Optional, Set, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import ColumnParallelLinear
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...utils.misc import filter_func_args
from ..layer_blocks.transformer_v2 import (
    TransformerV2AttType,
    TransformerV2CrossAttBlock,
    TransformerV2FeedForwardType,
    TransformerV2NormLayerType,
    TransformerV2SelfAttBlock,
)
from ..layers import RotaryPosEncoder
from ..layers.attention_v2 import ScaledDotProdAttV2
from ..utils import seq_lengths_to_cross_attn_mask
from .net_arch import NetArch


class QFormerV2(NetArch):
    """A Simplified Q-former not pretrained from any language model.
    It is called V2 because is based in V2 Transformer implementation, which based on LLAMA3 transformer implementation.

    Attributes:
        in_feats: input features dimension
        att_type: type of attention layer in [sdp, torch_sdp, flash_sdp]
        num_layers: transformer block repeats in each encoder stage
        hidden_dim: transformer block hidden features in each encoder stage
        num_heads: num. of attention heads
        num_kv_heads: num. of key, value attention heads when using GQA
        cross_att_freq: The frequency of adding cross-attention to the Transformer layers.
        att_dropout_rate: attention dropout rate
        att_bias: use bias in Linear layers of attention blocks
        ff_type: type of feed forward layer in [mlp, convnext]
        ff_dim_multiplier: number that multiplies the hidden dimension to get the inv. bottleneck dimension
        ff_multiple_of: the inv bottleneck dim has to be a multiple of this
        ff_kernel_size: kernel size when using convnext feed forward layer
        ff_dilation: dilation when using convnext feedforward layers
        ff_act: activation of feedforward layers
        ff_bias: use bias in Linear layers of feed forward blocks
        rope_in_self_att: use Rotary positional encoder or not positional encoder at all in self-attention
        rope_in_cross_att: use Rotary positional encoder or not positional encoder at all in cross-attention
        rope_theta: ROPE base theta
        rope_scale_freqs: scale ROPE frequencies when seq lenght is larger than the maximmum length of the original training sequences
        rope_update_max_seq_length: update the invernal ROPE variable that keeps track of the max seq length seen on training
        rope_original_max_seq_length: sets manually the max seq length seen in training for ROPE
        rope_scaling_factor: ROPE scaling factors
        rope_low_freq_factor: ROPE frequencies are not scaled for wavelengths < max_seq_length / self.low_freq_factor
        rope_high_freq_factor: ROPE frequencies are scaled by scaling for wavelengths > max_seq_length / self.high_freq_factor
        out_feats: features for output projection, if None, no output proj is done
        drop_path_rate: drop path rate
        norm_layer: type of norm layer in [layer-norm, rms-norm]
        norm_eps: eps for layer norms
        tied_layers: whether the encoder encoder layers are tied or not.
        multilayer_input: If True, input are hidden featues from several encoder layers.
        distribute_query_across_layers: splits the query into num_layers / num_cross_attention layers groups,
                                        the first group is used as input to the first cross-attention layer,
                                        the nth group is concantenated to input of the nth cross-attention layer,
        use_layer_idx_encoder: add a learned embedding based on the source layer index for cross-attention inputs.

        model_parallel: train with model parallel using fairscale tools

    """

    def __init__(
        self,
        in_feats: int,
        att_type: TransformerV2AttType = TransformerV2AttType.TORCH_SDP,
        num_layers: int = 3,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = None,
        cross_att_freq: int = 1,
        att_dropout_rate: float = 0.0,
        att_bias: bool = False,
        ff_type: TransformerV2FeedForwardType = TransformerV2FeedForwardType.MLP,
        ff_dim_multiplier: int = 4,
        ff_multiple_of: int = 256,
        ff_kernel_size: int = 7,
        ff_dilation: int = 1,
        ff_act: str = "silu",
        ff_bias: bool = False,
        rope_in_self_att: bool = False,
        rope_in_cross_att: bool = False,
        rope_theta: float = 50000,
        rope_scale_freqs: bool = True,
        rope_update_max_seq_length: bool = True,
        rope_original_max_seq_length: Optional[int] = None,
        rope_scaling_factor: float = 8,
        rope_low_freq_factor: float = 1,
        rope_high_freq_factor: float = 4,
        out_feats: Optional[int] = None,
        drop_path_rate: float = 0.0,
        norm_layer: TransformerV2NormLayerType = TransformerV2NormLayerType.LAYERNORM,
        norm_eps: float = 1e-5,
        tied_layers: bool = False,
        multilayer_input: bool = False,
        distribute_query_across_layers: bool = False,
        use_layer_idx_encoder: bool = False,
        model_parallel: bool = False,
    ):
        super().__init__()
        self.multilayer_input = multilayer_input
        if not isinstance(in_feats, int):
            raise TypeError("in_feats must be an int representing encoder feature dim")

        self.in_feats = in_feats
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.att_type = att_type
        self.att_dropout_rate = att_dropout_rate
        self.att_bias = att_bias
        self.cross_att_freq = cross_att_freq

        assert (
            num_layers // cross_att_freq
        ) * cross_att_freq == num_layers, (
            "num_layers should be multiple of cross_att_freq"
        )

        self.ff_type = ff_type
        self.ff_dim_multiplier = ff_dim_multiplier
        self.ff_multiple_of = ff_multiple_of
        self.ff_kernel_size = ff_kernel_size
        self.ff_dilation = ff_dilation
        self.ff_act = ff_act
        self.ff_bias = ff_bias

        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.norm_eps = norm_eps
        self._norm_layer = TransformerV2NormLayerType.to_class(norm_layer)

        self.tied_layers = tied_layers
        self.distribute_query_across_layers = distribute_query_across_layers
        if self.distribute_query_across_layers:
            self.num_query_groups = self.num_layers // self.cross_att_freq
        else:
            self.num_query_groups = 1

        self.rope_in_self_att = rope_in_self_att
        self.rope_in_cross_att = rope_in_cross_att
        self.rope_theta = rope_theta
        self.rope_scale_freqs = rope_scale_freqs
        self.rope_update_max_seq_length = rope_update_max_seq_length
        self.rope_original_max_seq_length = rope_original_max_seq_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor

        if rope_in_self_att or rope_in_cross_att:
            self.rope = RotaryPosEncoder(
                theta=rope_theta,
                scale_freqs=rope_scale_freqs,
                update_max_seq_length=rope_update_max_seq_length,
                original_max_seq_length=rope_original_max_seq_length,
                scaling_factor=rope_scaling_factor,
                low_freq_factor=rope_low_freq_factor,
                high_freq_factor=rope_high_freq_factor,
            )
        else:
            self.rope = None

        self.num_untied_layers = self.cross_att_freq if tied_layers else self.num_layers
        self.trans_blocks = nn.ModuleList()

        drop_rates = (
            torch.linspace(
                0.0,
                drop_path_rate,
                steps=max(self.num_untied_layers, 1),
                dtype=torch.float32,
            ).tolist()
            if drop_path_rate > 0.0
            else [0.0] * max(self.num_untied_layers, 1)
        )

        self.trans_blocks = nn.ModuleList()
        for i in range(self.num_untied_layers):
            drop_rate = drop_rates[i]
            if i % self.cross_att_freq == 0:
                block_i = TransformerV2CrossAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_feats=self.in_feats,
                    num_kv_heads=self.num_kv_heads,
                    ff_intermediate_feats=hidden_dim * self.ff_dim_multiplier,
                    ff_kernel_size=ff_kernel_size,
                    ff_dilation=ff_dilation,
                    ff_activation=self.ff_act,
                    ff_bias=self.ff_bias,
                    ff_multiple_of=self.ff_multiple_of,
                    att_dropout_rate=self.att_dropout_rate,
                    att_bias=self.att_bias,
                    rope=self.rope,
                    rope_in_self_att=rope_in_self_att,
                    rope_in_cross_att=rope_in_cross_att,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    drop_path_rate=drop_rate,
                    model_parallel=model_parallel,
                )
            else:
                block_i = TransformerV2SelfAttBlock(
                    att_type=self.att_type,
                    ff_type=self.ff_type,
                    num_feats=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    ff_intermediate_feats=hidden_dim * self.ff_dim_multiplier,
                    ff_kernel_size=ff_kernel_size,
                    ff_dilation=ff_dilation,
                    ff_activation=self.ff_act,
                    ff_bias=self.ff_bias,
                    ff_multiple_of=self.ff_multiple_of,
                    att_dropout_rate=self.att_dropout_rate,
                    att_bias=self.att_bias,
                    rope=self.rope,
                    norm_layer=self._norm_layer,
                    norm_eps=self.norm_eps,
                    drop_path_rate=drop_rate,
                    model_parallel=model_parallel,
                )
            self.trans_blocks.append(block_i)

        self.model_parallel = model_parallel

        # head feature block
        self.out_norm = self._norm_layer(hidden_dim, eps=norm_eps)
        if out_feats is not None and out_feats > 0:
            self.out_feats = out_feats
            if model_parallel:
                self.out_proj = ColumnParallelLinear(
                    hidden_dim,
                    out_feats,
                    bias=False,
                )
            else:
                self.out_proj = nn.Linear(hidden_dim, out_feats, bias=False)
        else:
            self.out_feats = None

        self.use_layer_idx_encoder = use_layer_idx_encoder
        if use_layer_idx_encoder:
            assert (
                self.tied_layers
            ), "layer idx encoder can be used only with tied layers"
            self.layer_idx_encoder = nn.Embedding(
                self.num_layers // self.cross_att_freq, self.in_feats
            )
        else:
            self.layer_idx_encoder = None

        self._init_weights()

    def _init_weights(self):
        """Initialize all learnable weights following BLIP-2 defaults."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def _compute_out_size(self, in_size):
        """Return the temporal dimension after processing the query stream.

        Args:
            in_size: Temporal dimension of the query input.

        """
        out_size = in_size
        return out_size

    def in_feats_shape(self):
        """Describe the expected `(batch, time, channels)` shape for encoder features."""
        return (None, None, self.in_feats)

    def query_shape(self):
        """Describe the `(batch, time, hidden_dim)` shape for the query tokens."""
        return (None, None, self.hidden_dim)

    def out_shape(self, query_shape=None):
        """Infer the output shape for a given query input shape.

        Args:
            query_shape: Optional `(batch, time, channels)` tuple describing the query input.

        """
        out_channels = self.out_feats if self.out_feats is not None else self.hidden_dim
        if query_shape is None:
            return (None, None, out_channels)

        assert len(query_shape) == 3, "query_shape must be (batch, time, channels)"
        if query_shape[1] is None:
            T = None
        else:
            T = self._compute_out_size(query_shape[1])

        return (query_shape[0], T, out_channels)

    @property
    def out_dim(self):
        """Return the output feature dimension after the Q-Former."""
        return self.out_feats if self.out_feats is not None else self.hidden_dim

    def out_channels(self):
        """Return the output feature dimension after the Q-Former."""
        return self.out_feats if self.out_feats is not None else self.hidden_dim

    @property
    def output_is_normalized(self):
        return True if self.out_feats is None else False

    def forward(
        self,
        query_embeds: torch.Tensor,
        feats: Union[torch.Tensor, List[torch.Tensor]],
        feats_lengths: Optional[
            Union[torch.Tensor, List[Optional[torch.Tensor]]]
        ] = None,
        start_pos: int = 0,
    ):
        """Dispatch the forward pass to the appropriate pathway.

        Args:
            query_embeds: Query tensor of shape `(batch, query_len, hidden_dim)`.
            feats: Encoder representations; either a single tensor `(batch, seq_len, in_feats)`
                or a list of tensors when `multilayer_input` is enabled.
            feats_lengths: Optional sequence-length tensor(s) used to build attention masks.
            start_pos: Starting rotary/cache position for cross-attention.

        """
        if self.multilayer_input:
            return self.forward_multilayer_input(
                query_embeds, feats, feats_lengths, start_pos=start_pos
            )
        else:
            return self.forward_singlelayer_input(
                query_embeds, feats, feats_lengths, start_pos=start_pos
            )

    def forward_singlelayer_input(
        self,
        query_embeds: torch.Tensor,
        feats: Union[torch.Tensor, List[torch.Tensor]],
        feats_lengths: Optional[
            Union[torch.Tensor, List[Optional[torch.Tensor]]]
        ] = None,
        start_pos: int = 0,
    ):
        """Process a single stream of encoder features through the Q-Former.

        Args:
            query_embeds: Query tensor `(batch, query_len, hidden_dim)`.
            feats: Encoder features `(batch, seq_len, in_feats)` or a singleton list containing them.
            feats_lengths: Optional sequence-length tensor(s) aligned with `feats`.
            start_pos: Starting rotary/cache position for cross-attention.

        """
        query_dtype = query_embeds.dtype
        query_device = query_embeds.device
        if isinstance(feats, list):
            if len(feats) != 1:
                raise ValueError(
                    "forward_singlelayer_input expects a single feature tensor when feats is a list"
                )
            feats_tensor = feats[0]
            if isinstance(feats_lengths, list):
                if len(feats_lengths) != 1:
                    raise ValueError(
                        "forward_singlelayer_input expects a single sequence length entry when feats is a list"
                    )
                feats_lengths = feats_lengths[0]
        else:
            feats_tensor = feats

        if feats_lengths is not None and isinstance(feats_lengths, list):
            raise ValueError(
                "feats_lengths must be a tensor or None when feats is a single tensor"
            )

        if not torch.all(torch.isfinite(feats_tensor)):
            logging.warning("non-finite feats=%f", torch.mean(feats_tensor))

        if self.distribute_query_across_layers:
            query_length = query_embeds.size(1)
            if query_length % self.num_query_groups != 0:
                raise ValueError(
                    "query length must be divisible by num_query_groups when distribute_query_across_layers is enabled"
                )
            query_embeds = torch.split(
                query_embeds, query_embeds.size(1) // self.num_query_groups, dim=1
            )
            hidden_feats = query_embeds[0]
            mask_query_length = query_length
        else:
            hidden_feats = query_embeds
            mask_query_length = query_embeds.size(1)

        if feats_lengths is not None:
            feats_mask = seq_lengths_to_cross_attn_mask(
                None,
                feats_lengths,
                max_query_length=mask_query_length,
                max_kv_length=feats_tensor.size(1),
                dtype=query_dtype,
                device=query_device,
                none_if_all_max=True,
            )
        else:
            feats_mask = None

        for i in range(self.num_layers):
            layer_idx = i % self.num_untied_layers
            if i % self.cross_att_freq == 0:
                if self.distribute_query_across_layers and i > 0:
                    hidden_feats = torch.cat(
                        (hidden_feats, query_embeds[i // self.cross_att_freq]), dim=1
                    )

                if self.use_layer_idx_encoder:
                    layer_idx_embeds = self.layer_idx_encoder.weight[
                        i // self.cross_att_freq
                    ].view(1, 1, -1)
                    cur_feats = feats_tensor + layer_idx_embeds
                else:
                    cur_feats = feats_tensor
                hidden_feats = self.trans_blocks[layer_idx](
                    hidden_feats,
                    x_kv=cur_feats,
                    x_kv_mask=feats_mask,
                    start_pos_kv=start_pos,
                )
            else:
                hidden_feats = self.trans_blocks[layer_idx](hidden_feats)

            if not torch.all(torch.isfinite(hidden_feats)):
                logging.warning(
                    "non-finite x-enc-%d-avg=%f", i, torch.mean(hidden_feats)
                )

        out_feats = self.out_norm(hidden_feats)

        if self.out_feats is not None:
            out_feats = self.out_proj(out_feats)

        if not torch.all(torch.isfinite(out_feats)):
            logging.warning("non-finite x-out-avg=%f", torch.mean(out_feats))
        return out_feats

    def forward_multilayer_input(
        self,
        query_embeds: torch.Tensor,
        feats: List[torch.Tensor],
        feats_lengths: Optional[List[Optional[torch.Tensor]]] = None,
        start_pos: int = 0,
    ):
        """Process encoder hidden states from multiple layers one cross-att step at a time.

        Args:
            query_embeds: Query tensor `(batch, query_len, hidden_dim)`.
            feats: List of encoder feature tensors, one per cross-attention layer.
            feats_lengths: Optional list matching `feats` with sequence-length tensors.
            start_pos: Starting rotary/cache position for cross-attention.

        """
        assert (
            len(feats) == self.num_layers // self.cross_att_freq
        ), "feats must match num cross-attention layers"
        if feats_lengths is not None:
            assert len(feats) == len(
                feats_lengths
            ), "feats_lengths must align with feats for multilayer input"
        for idx, feat in enumerate(feats):
            if not torch.all(torch.isfinite(feat)):
                logging.warning("non-finite x-in-%d-avg=%f", idx, torch.mean(feat))
                break

        query_dtype = query_embeds.dtype
        query_device = query_embeds.device
        if self.distribute_query_across_layers:
            query_length = query_embeds.size(1)
            if query_length % self.num_query_groups != 0:
                raise ValueError(
                    "query length must be divisible by num_query_groups when distribute_query_across_layers is enabled"
                )
            query_embeds = torch.split(
                query_embeds, query_embeds.size(1) // self.num_query_groups, dim=1
            )
            hidden_feats = query_embeds[0]
            mask_query_length = query_length
        else:
            hidden_feats = query_embeds
            mask_query_length = query_embeds.size(1)

        last_max_feats_lengths = 0
        feats_mask = None
        for i in range(self.num_layers):
            layer_idx = i % self.num_untied_layers
            if i % self.cross_att_freq == 0:
                if self.distribute_query_across_layers and i > 0:
                    hidden_feats = torch.cat(
                        (hidden_feats, query_embeds[i // self.cross_att_freq]), dim=1
                    )

                feats_idx = i // self.cross_att_freq
                cur_feats = feats[feats_idx]
                if (
                    feats_lengths is not None
                    and feats_lengths[feats_idx] is not None
                    and cur_feats.size(1) != last_max_feats_lengths
                ):
                    last_max_feats_lengths = cur_feats.size(1)
                    feats_mask = seq_lengths_to_cross_attn_mask(
                        None,
                        feats_lengths[feats_idx],
                        max_query_length=mask_query_length,
                        max_kv_length=last_max_feats_lengths,
                        dtype=query_dtype,
                        device=query_device,
                        none_if_all_max=True,
                    )

                if self.use_layer_idx_encoder:
                    layer_idx_embeds = self.layer_idx_encoder.weight[feats_idx].view(
                        1, 1, -1
                    )
                    cur_feats = cur_feats + layer_idx_embeds

                hidden_feats = self.trans_blocks[layer_idx](
                    hidden_feats,
                    x_kv=cur_feats,
                    x_kv_mask=feats_mask,
                    start_pos_kv=start_pos,
                )
            else:
                hidden_feats = self.trans_blocks[layer_idx](hidden_feats)

            if not torch.all(torch.isfinite(hidden_feats)):
                logging.warning(
                    "non-finite x-enc-%d-avg=%f", i, torch.mean(hidden_feats)
                )

        out_feats = self.out_norm(hidden_feats)

        if self.out_feats is not None:
            out_feats = self.out_proj(out_feats)

        if not torch.all(torch.isfinite(out_feats)):
            logging.warning("non-finite x-out-avg=%f", torch.mean(out_feats))
        return out_feats

    def get_config(self, no_class_name: bool = False) -> dict:
        """Return a JSON-serializable snapshot of the constructor arguments."""
        config = {
            "in_feats": self.in_feats,
            "att_type": self.att_type,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "cross_att_freq": self.cross_att_freq,
            "att_dropout_rate": self.att_dropout_rate,
            "att_bias": self.att_bias,
            "ff_type": self.ff_type,
            "ff_dim_multiplier": self.ff_dim_multiplier,
            "ff_multiple_of": self.ff_multiple_of,
            "ff_kernel_size": self.ff_kernel_size,
            "ff_dilation": self.ff_dilation,
            "ff_act": self.ff_act,
            "ff_bias": self.ff_bias,
            "rope_in_self_att": self.rope_in_self_att,
            "rope_in_cross_att": self.rope_in_cross_att,
            "rope_theta": self.rope_theta,
            "rope_scale_freqs": self.rope_scale_freqs,
            "rope_update_max_seq_length": self.rope_update_max_seq_length,
            "rope_original_max_seq_length": self.rope_original_max_seq_length,
            "rope_scaling_factor": self.rope_scaling_factor,
            "rope_low_freq_factor": self.rope_low_freq_factor,
            "rope_high_freq_factor": self.rope_high_freq_factor,
            "out_feats": self.out_feats,
            "norm_eps": self.norm_eps,
            "drop_path_rate": self.drop_path_rate,
            "norm_layer": self.norm_layer,
            "tied_layers": self.tied_layers,
            "multilayer_input": self.multilayer_input,
            "distribute_query_across_layers": self.distribute_query_across_layers,
            "use_layer_idx_encoder": self.use_layer_idx_encoder,
            "model_parallel": self.model_parallel,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, override_dropouts: bool, drop_path_rate: float, att_dropout_rate: float
    ):
        """Optionally override dropout hyperparameters during fine-tuning.

        Args:
            override_dropouts: If ``True``, propagate the supplied dropout values.
            drop_path_rate: New global stochastic depth probability.
            att_dropout_rate: New attention dropout probability.

        """
        if override_dropouts:
            logging.info("chaning convnext1d dropouts")
            self.change_dropouts(drop_path_rate, att_dropout_rate)

    def change_dropouts(self, drop_path_rate: float, att_dropout_rate: float):
        """Propagate new dropout values through all attention and DropPath modules.

        Args:
            drop_path_rate: Target stochastic depth probability.
            att_dropout_rate: Target attention dropout probability.

        """
        from ..layers import DropPath1d

        for module in self.modules():
            if isinstance(module, DropPath1d):
                module.p *= drop_path_rate / self.drop_path_rate

            if isinstance(module, ScaledDotProdAttV2):
                module.dropout_rate = att_dropout_rate

        self.drop_path_rate = drop_path_rate

    @staticmethod
    def filter_args(**kwargs):
        """Filter keyword arguments down to those accepted by the constructor.

        Args:
            **kwargs: Candidate keyword arguments.

        """
        return filter_func_args(QFormerV2.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ):
        """Register constructor arguments with a JSONArgParse parser.

        Args:
            parser: Target `ArgumentParser` instance.
            prefix: Optional nested argument namespace.
            skip: Optional set of field names to exclude.

        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if skip is None:
            skip = set([])

        if "in_feats" not in skip:
            parser.add_argument("--in-feats", type=int, help="input features dimension")

        parser.add_argument(
            "--att-type",
            default=TransformerV2AttType.TORCH_SDP.value,
            choices=TransformerV2AttType.choices(),
            help="type of attention layer in [sdp, torch_sdp, flash_sdp]",
        )
        parser.add_argument(
            "--num-layers",
            default=3,
            type=int,
            help="transformer block repeats in each encoder stage",
        )
        parser.add_argument(
            "--hidden-dim",
            default=768,
            type=int,
            help="transformer block hidden features in each encoder stage",
        )
        parser.add_argument(
            "--num-heads", default=12, type=int, help="num of attention heads"
        )
        parser.add_argument(
            "--num-kv-heads",
            default=None,
            type=int,
            help="num. of key, value attention heads when using GQA",
        )
        parser.add_argument(
            "--cross-att-freq",
            default=1,
            type=int,
            help="The frequency of adding cross-attention to the Transformer layers.",
        )
        parser.add_argument(
            "--att-dropout-rate", default=0.0, type=float, help="attention dropout rate"
        )
        parser.add_argument(
            "--att-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of attention blocks",
        )
        parser.add_argument(
            "--ff-type",
            default=TransformerV2FeedForwardType.MLP.value,
            choices=TransformerV2FeedForwardType.choices(),
            help="type of feed forward layer in [mlp, convnext]",
        )
        parser.add_argument(
            "--ff-dim-multiplier",
            default=4,
            type=int,
            help="number that multiplies the hidden dimension to get the inv. bottleneck dimension",
        )
        parser.add_argument(
            "--ff-multiple-of",
            default=256,
            type=int,
            help="the inv bottleneck dim has to be a multiple of this",
        )
        parser.add_argument(
            "--ff-kernel-size",
            default=7,
            type=int,
            help="kernel size when using convnext feed forward layer",
        )
        parser.add_argument(
            "--ff-dilation",
            default=1,
            type=int,
            help="dilation when using convnext feedforward layers",
        )
        parser.add_argument(
            "--ff-act", default="silu", help="activation of feedforward layers"
        )
        parser.add_argument(
            "--ff-bias",
            default=False,
            action=ActionYesNo,
            help="use bias in Linear layers of feed forward blocks",
        )
        parser.add_argument(
            "--rope-theta", default=50000, type=float, help="ROPE base theta"
        )
        parser.add_argument(
            "--rope-scale-freqs",
            default=True,
            action=ActionYesNo,
            help="scale ROPE frequencies when seq lenght is larger than the maximmum length of the original training sequences",
        )
        parser.add_argument(
            "--rope-update-max-seq-length",
            default=True,
            action=ActionYesNo,
            help="update the invernal ROPE variable that keeps track of the max seq length seen on training",
        )
        parser.add_argument(
            "--rope-original-max-seq-length",
            default=None,
            type=int,
            help="sets manually the max seq length seen in training for ROPE",
        )
        parser.add_argument(
            "--rope-scaling-factor", default=8, type=float, help="ROPE scaling factors"
        )
        parser.add_argument(
            "--rope-low-freq-factor",
            default=1,
            type=float,
            help="ROPE frequencies are not scaled for wavelengths < max_seq_length / self.low_freq_factor",
        )
        parser.add_argument(
            "--rope-high-freq-factor",
            default=4,
            type=float,
            help="ROPE frequencies are scaled by scaling for wavelengths > max_seq_length / self.high_freq_factor",
        )
        parser.add_argument(
            "--rope-in-self-att",
            default=False,
            action=ActionYesNo,
            help="use Rotary positional encoder or not positional encoder at all in self-attention",
        )
        parser.add_argument(
            "--rope-in-cross-att",
            default=False,
            action=ActionYesNo,
            help="use Rotary positional encoder or not positional encoder at all in cross-attention",
        )
        parser.add_argument(
            "--distribute-query-across-layers",
            default=False,
            action=ActionYesNo,
            help="""splits the query into num_layers / num_cross_attention layers groups, 
                    the first group is used as input to the first cross-attention layer,
                    the nth group is concantenated to input of the nth cross-attention layer""",
        )
        parser.add_argument(
            "--tied-layers",
            default=False,
            action=ActionYesNo,
            help="whether the encoder encoder layers are tied or not.",
        )
        parser.add_argument(
            "--multilayer-input",
            default=False,
            action=ActionYesNo,
            help="Input are hidden featues from several encoder layers",
        )
        parser.add_argument(
            "--use-layer-idx-encoder",
            default=False,
            action=ActionYesNo,
            help="add a learned embedding of the encoder layer index to cross-attention inputs",
        )
        parser.add_argument(
            "--out-feats",
            default=None,
            type=int,
            help="features for ouptut projection, if None, no output proj is done",
        )
        parser.add_argument(
            "--drop-path-rate", default=0.0, type=float, help="drop path rate"
        )
        parser.add_argument(
            "--norm-layer",
            default=TransformerV2NormLayerType.LAYERNORM.value,
            choices=TransformerV2NormLayerType.choices(),
            help="type of norm layer in [layer-norm, rms-norm]",
        )
        parser.add_argument(
            "--norm-eps", default=1e-5, type=float, help="eps for layer norms"
        )
        parser.add_argument(
            "--model-parallel",
            default=False,
            action=ActionYesNo,
            help="train with model parallel using fairscale tools",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        """Filter keyword arguments to those handled by `change_config`.

        Args:
            **kwargs: Candidate keyword arguments.

        """
        return filter_func_args(QFormerV2.change_config, kwargs)

    @staticmethod
    def add_finetune_args(parser, prefix=None, skip=set([])):
        """Register fine-tuning specific overrides with a JSONArgParse parser.

        Args:
            parser: Target `ArgumentParser` instance.
            prefix: Optional nested argument namespace.
            skip: Optional set of field names to exclude.

        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        try:
            parser.add_argument(
                "--override-dropouts",
                default=False,
                action=ActionYesNo,
                help=(
                    "whether to use the dropout probabilities passed in the "
                    "arguments instead of the defaults in the pretrained model."
                ),
            )
        except:
            pass

        try:
            parser.add_argument(
                "--drop-path-rate",
                default=0,
                type=float,
                help="layer drop probability",
            )
        except:
            pass

        try:
            parser.add_argument(
                "--att-dropout-rate",
                default=0,
                type=float,
                help="attention layers dropout rate",
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

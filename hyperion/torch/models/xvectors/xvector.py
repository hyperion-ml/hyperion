"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HyperDataClass
from ....utils.misc import filter_func_args
from ...hyper_torch_model import HyperTorchModel
from ...layer_blocks import TDNNBlock
from ...layers import GlobalPool1dFactory as PF
from ...narchs import ClassifHead, DINOHead, ProjHead, TorchNALoader
from ...utils import eval_nnet_by_chunks, scale_seq_lengths


class XVectorHeadType(str, Enum):
    XVECTOR = "x-vector"
    DINO = "dino"

    @staticmethod
    def choices() -> List[str]:
        """Return the valid head type strings.

        Returns:
            List of valid head type strings.
        """
        return [o.value for o in XVectorHeadType]


@dataclass
class XVectorOutput(HyperDataClass):
    loss: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    xvector: Optional[torch.Tensor]
    h_enc: Optional[List[torch.Tensor]] = None
    h_classif: Optional[List[torch.Tensor]] = None
    h_feats: Optional[List[torch.Tensor]] = None


class XVector(HyperTorchModel):
    """Base class for x-vector style models.

    Attributes:
        encoder_net: Encoder network that produces frame-level features.
        in_feats: Input feature dimension, if known.
        proj: Optional projection layer between encoder and pooling.
        proj_feats: Output dimension of ``proj`` when present.
        pool_net: Temporal pooling module.
        classif_net: Classification head.
        proj_head_net: Optional projection head used by DINO-style heads.
        head_type: Classification head type.
        embed_dim: Output embedding size of the classification head.
        num_embed_layers: Number of hidden layers in the head.
        dropout_rate: Dropout rate applied in the head.
    """

    def __init__(
        self,
        encoder_net: Any,
        num_classes: int,
        pool_net: Union[str, Dict[str, Any], nn.Module] = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[str, Dict[str, Any], Callable[..., nn.Module]] = {
            "name": "relu",
            "inplace": True,
        },
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        head_norm_layer: Optional[Union[str, Callable[..., nn.Module]]] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        head_use_norm: bool = True,
        head_use_in_norm: bool = False,
        head_hid_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        proj_head_use_norm: bool = True,
        proj_head_norm_before: bool = True,
        dropout_rate: float = 0,
        embed_layer: int = 0,
        in_feats: Optional[int] = None,
        proj_feats: Optional[int] = None,
        head_type: Union[str, XVectorHeadType] = XVectorHeadType.XVECTOR,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Build an x-vector model.

        Args:
            encoder_net: Encoder network instance.
            num_classes: Number of output classes.
            pool_net: Pooling configuration or module.
            embed_dim: X-vector embedding dimension.
            num_embed_layers: Number of hidden layers in the classification head.
            hid_act: Hidden activation configuration.
            loss_type: Classification loss type.
            cos_scale: Scaling factor for angular-margin losses.
            margin: Margin for angular-margin losses.
            margin_warmup_epochs: Margin warmup duration in epochs.
            intertop_k: InterTopK penalty parameter.
            intertop_margin: InterTopK margin parameter.
            num_subcenters: Number of subcenters for subcenter losses.
            norm_layer: Normalization layer configuration.
            head_norm_layer: Normalization layer configuration for the head.
            use_norm: Whether to use normalization in auxiliary blocks.
            norm_before: Whether normalization is applied before activation.
            head_use_norm: Whether to use normalization in the head.
            head_use_in_norm: Whether to normalize head inputs.
            head_hid_dim: Hidden dimension for DINO heads.
            head_bottleneck_dim: Bottleneck dimension for DINO heads.
            proj_head_use_norm: Whether to normalize the projection head.
            proj_head_norm_before: Whether projection head normalization happens before activation.
            dropout_rate: Dropout rate used in the head.
            embed_layer: Head layer index used for embeddings.
            in_feats: Input feature dimension, if known.
            proj_feats: Optional projection feature dimension after the encoder.
            head_type: Classification head type.
            bias_weight_decay: Optional weight decay for bias parameters.
        """
        super().__init__(bias_weight_decay=bias_weight_decay)

        # encoder network
        self.encoder_net = encoder_net

        # infer input and output shapes of encoder network
        in_shape = self.encoder_net.in_shape()
        if len(in_shape) == 3:
            # encoder based on 1d conv or transformer
            in_feats = in_shape[1]
            out_shape = self.encoder_net.out_shape(in_shape)
            enc_feats = out_shape[1]
            if enc_feats is None:
                # encoder gives (B, T, F)
                enc_feats = out_shape[2]

        elif len(in_shape) == 4:
            # encoder based in 2d convs
            assert (
                in_feats is not None
            ), "in_feats dimension must be given to calculate pooling dimension"
            in_shape = list(in_shape)
            in_shape[2] = in_feats
            out_shape = self.encoder_net.out_shape(tuple(in_shape))
            enc_feats = out_shape[1] * out_shape[2]

        self.in_feats = in_feats

        logging.info("encoder input shape={}".format(in_shape))
        logging.info("encoder output shape={}".format(out_shape))

        # add projection network to link encoder and pooling layers if proj_feats is not None
        self.proj = None
        self.proj_feats = proj_feats
        if proj_feats is not None:
            logging.info(
                "adding projection layer after encoder with in/out size %d -> %d",
                enc_feats,
                proj_feats,
            )

            self.proj = TDNNBlock(
                enc_feats, proj_feats, kernel_size=1, activation=None, use_norm=use_norm
            )

        # create pooling network
        # infer output dimension of pooling which is input dim for classification head
        if proj_feats is None:
            self.pool_net = self._make_pool_net(pool_net, enc_feats)
            pool_feats = int(enc_feats * self.pool_net.size_multiplier)
        else:
            self.pool_net = self._make_pool_net(pool_net, proj_feats)
            pool_feats = int(proj_feats * self.pool_net.size_multiplier)

        logging.info("infer pooling dimension %d", pool_feats)

        # if head_norm_layer is none we use the global norm_layer
        if head_norm_layer is None and norm_layer is not None:
            if norm_layer in ("instance-norm", "instance-norm-affine"):
                head_norm_layer = "batch-norm"
            else:
                head_norm_layer = norm_layer

        # create classification head
        logging.info("making classification head net")
        self.embed_dim = embed_dim
        self.num_embed_layers = num_embed_layers
        self.head_type = head_type
        self.hid_act = hid_act
        self.norm_layer = norm_layer
        self.use_norm = use_norm
        self.norm_before = norm_before
        self.head_use_in_norm = head_use_in_norm
        self.head_use_norm = head_use_norm
        self.head_norm_layer = head_norm_layer
        self.head_hid_dim = head_hid_dim
        self.head_bottleneck_dim = head_bottleneck_dim
        self.proj_head_use_norm = proj_head_use_norm
        self.proj_head_norm_before = proj_head_norm_before
        self.dropout_rate = dropout_rate
        self.embed_layer = embed_layer
        if self.head_type == XVectorHeadType.XVECTOR:
            self.proj_head_net = None
            self.classif_net = ClassifHead(
                pool_feats,
                num_classes,
                embed_dim=embed_dim,
                num_embed_layers=num_embed_layers,
                hid_act=hid_act,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                norm_layer=head_norm_layer,
                use_norm=head_use_norm,
                norm_before=norm_before,
                dropout_rate=dropout_rate,
                use_in_norm=head_use_in_norm,
            )
        elif self.head_type == XVectorHeadType.DINO:
            self.proj_head_net = ProjHead(
                pool_feats,
                embed_dim,
                norm_layer=head_norm_layer,
                use_norm=proj_head_use_norm,
                norm_before=proj_head_norm_before,
            )
            self.classif_net = DINOHead(
                embed_dim,
                num_classes,
                hid_feats=head_hid_dim,
                bottleneck_feats=head_bottleneck_dim,
                num_hid_layers=num_embed_layers,
                hid_act=hid_act,
                output_type=loss_type,
                norm_layer=head_norm_layer,
                use_norm=head_use_norm,
                norm_before=norm_before,
                dropout_rate=dropout_rate,
                use_in_norm=head_use_in_norm,
            )

    @property
    def pool_feats(self):
        """Return the pooling feature dimension.

        Returns:
            Feature dimension consumed by the classification head.
        """
        if self.proj_head_net is None:
            return self.classif_net.in_feats
        else:
            return self.proj_head_net.in_feats

    @property
    def num_classes(self):
        """Return the number of output classes.

        Returns:
            Number of output classes.
        """
        return self.classif_net.num_classes

    @property
    def cos_scale(self):
        """Return the angular-margin scale.

        Returns:
            Scaling factor used by the head.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.cos_scale
        elif self.head_type == XVectorHeadType.DINO:
            return 1
        else:
            raise ValueError

    @property
    def margin(self):
        """Return the current margin value.

        Returns:
            Margin used by the head.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.margin
        else:
            return 0.0

    @property
    def margin_warmup_epochs(self):
        """Return the margin warmup duration.

        Returns:
            Margin warmup duration in epochs.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.margin_warmup_epochs
        else:
            return 0

    @property
    def intertop_k(self):
        """Return the InterTopK parameter.

        Returns:
            InterTopK `k` value.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.intertop_k
        else:
            return 0

    @property
    def intertop_margin(self):
        """Return the InterTopK margin.

        Returns:
            InterTopK margin value.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.intertop_margin
        else:
            return 0.0

    @property
    def num_subcenters(self):
        """Return the number of subcenters.

        Returns:
            Number of subcenters used by the head.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.num_subcenters
        else:
            return 0

    @property
    def loss_type(self):
        """Return the configured loss type.

        Returns:
            Loss type string.
        """
        if self.head_type == XVectorHeadType.XVECTOR:
            return self.classif_net.loss_type
        elif self.head_type == XVectorHeadType.DINO:
            return self.classif_net.output_type
        else:
            raise ValueError()

    def _make_pool_net(
        self,
        pool_net: Union[str, Dict[str, Any], nn.Module],
        enc_feats: Optional[int] = None,
    ) -> nn.Module:
        """Makes the pooling block

        Args:
            pool_net: Pooling configuration string, dictionary, or module.
            enc_feats: Input feature dimension from the encoder.

        Returns:
            Pooling module.
        """
        if isinstance(pool_net, str):
            pool_net = {"pool_type": pool_net}

        if isinstance(pool_net, dict):
            if enc_feats is not None:
                pool_net["in_feats"] = enc_feats

            return PF.create(**pool_net)
        elif isinstance(pool_net, nn.Module):
            return pool_net
        else:
            raise Exception("Invalid pool_net argument")

    def update_loss_margin(self, epoch: int) -> None:
        """Updates the value of the margin in AAM/AM-softmax losses
           given the epoch number

        Args:
            epoch: Epoch that is about to start.
        """
        self.classif_net.update_margin(epoch)

    def _pre_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape inputs to match the encoder layout.

        Args:
            x: Input feature tensor.

        Returns:
            Tensor formatted for the encoder.
        """
        if self.encoder_net.in_dim() == 4 and x.dim() == 3:
            x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
        return x

    def _post_enc(
        self,
        x: torch.Tensor,
        in_lengths: Optional[torch.Tensor] = None,
        max_in_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Undo encoder-specific layout changes and scale sequence lengths.

        Args:
            x: Encoder output tensor.
            in_lengths: Input sequence lengths.
            max_in_length: Maximum input length before encoding.

        Returns:
            Tuple of post-processed features and output lengths.
        """
        if self.encoder_net.out_dim() == 4:
            x = x.view(x.size(0), -1, x.size(-1))

        if self.proj is not None:
            x = self.proj(x)

        if in_lengths is not None:
            out_lengths = scale_seq_lengths(in_lengths, x.size(-1), max_in_length)
        else:
            out_lengths = None

        return x, out_lengths

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_enc_layers: Optional[Sequence[int]] = None,
        return_classif_layers: Optional[Sequence[int]] = None,
        return_logits: bool = True,
    ) -> XVectorOutput:
        """Forward function. If returns the logits posteriors of the classes.
        It can also returns the hidden representations in the encoder and
        classification head. In this case the ouput variable is a dictionary.

        Args:
          x: input features tensor with shape=(batch, in_feats, time).
          x_lengths: time lengths of the features with shape=(batch,).
          y: target classes torch.long tensor with shape=(batch,).
          return_enc_layers: list of integers indicating, which encoder layers
                             we should return. If None, no encoder layers are returned.
          return_classif_layers: list of integers indicating, which classification head layers
                             we should return. If None, no head layers are returned.
          return_logits: if True, it adds the logits to the output dictionary.
        Returns:
          Tensor with class logits with shape=(batch, num_classes) or
          Dictionary with "logits", "h_enc" (list of hidden encoder layers),
          "h_classif" (list hidden classification head layers).
        """

        if return_enc_layers is None and return_classif_layers is None:
            return self.forward_logits(x, x_lengths, y)

        return self.forward_hid_feats(
            x, x_lengths, y, return_enc_layers, return_classif_layers, return_logits
        )

    def forward_logits(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> XVectorOutput:
        """Forward function

        Args:
            x: Input features tensor with shape=(batch, in_feats, time).
            x_lengths: Time lengths of the features with shape=(batch,).
            y: Target classes tensor with shape=(batch,).

        Returns:
            Output dataclass with class logits and optional embedding.
        """
        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        x = self.encoder_net(x, x_lengths=x_lengths)
        if isinstance(x, tuple):
            x = x[0]

        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
        p = self.pool_net(x, x_lengths=x_lengths)
        if not torch.all(torch.isfinite(p)):
            logging.warning("non-finite p-avg=%f", torch.mean(p))
        xvector = None
        if self.proj_head_net is not None:
            p = self.proj_head_net(p)
            xvector = p

        logits = self.classif_net(p, y)
        if not torch.all(torch.isfinite(logits)):
            logging.warning("non-finite y-avg=%f", torch.mean(logits))
        # return logits
        output = XVectorOutput(None, logits, xvector)
        return output

    def forward_hid_feats(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_enc_layers: Optional[Sequence[int]] = None,
        return_classif_layers: Optional[Sequence[int]] = None,
        return_logits: bool = False,
    ) -> XVectorOutput:
        """forwards hidden representations in the x-vector network

        Args:
            x: Input features tensor with shape=(batch, in_feats, time).
            x_lengths: Time lengths of the features with shape=(batch,).
            y: Target classes tensor with shape=(batch,).
            return_enc_layers: list of integers indicating, which encoder layers
                we should return. If None, no encoder layers are returned.
            return_classif_layers: list of integers indicating, which classification head layers
                we should return. If None, no head layers are returned.
            return_logits: if True, it adds the logits to the output dictionary.
        Returns:
            Output dataclass with hidden features and optional logits.
        """
        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        h_enc, x = self.encoder_net.forward_hid_feats(
            x, x_lengths=x_lengths, layers=return_enc_layers, return_output=True
        )
        output = {"h_enc": h_enc}
        if not return_logits and return_classif_layers is None:
            output = XVectorOutput(None, None, None, h_enc, None)
            return output

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        if self.proj_head_net is not None:
            p = self.proj_head_net(p)
        h_classif, y_pred = self.classif_net.forward_hid_feats(
            p, y, return_classif_layers, return_logits=return_logits
        )
        if h_classif:
            xvector = h_classif[0]
        else:
            xvector = None

        output = XVectorOutput(None, y_pred, xvector, h_enc, h_classif)
        return output

    # def forward_hid_feats(
    #     self,
    #     x,
    #     x_lengths=None,
    #     y=None,
    #     return_enc_layers=None,
    #     return_classif_layers=None,
    #     return_logits=False,
    # ):
    #     """forwards hidden representations in the x-vector network

    #     Args:
    #       x: input features tensor with shape=(batch, in_feats, time).
    #       x_lengths: time lengths of the features with shape=(batch,).
    #       y: target classes torch.long tensor with shape=(batch,).
    #       return_enc_layers: list of integers indicating, which encoder layers
    #                          we should return. If None, no encoder layers are returned.
    #       return_enc_layers: list of integers indicating, which classification head layers
    #                          we should return. If None, no head layers are returned.
    #       return_logits: if True, it adds the logits to the output dictionary.
    #     Returns:
    #       Dictionary with "logits", "h_enc" (list of hidden encoder layers),
    #       "h_classif" (list hidden classification head layers).
    #     """
    #     max_in_length = x.size(-1)
    #     x = self._pre_enc(x)
    #     h_enc, x = self.encoder_net.forward_hid_feats(
    #         x, return_enc_layers, return_output=True
    #     )
    #     output = {"h_enc": h_enc}
    #     if not return_logits and return_classif_layers is None:
    #         return output

    #     x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
    #     p = self.pool_net(x, x_lengths=x_lengths)
    #     if self.proj_head_net is not None:
    #         p = self.proj_head_net(p)
    #     h_classif = self.classif_net.forward_hid_feats(
    #         p, y, return_classif_layers, return_logits=return_logits
    #     )
    #     if return_logits:
    #         h_classif, y_pred = h_classif
    #         output["h_classif"] = h_classif
    #         output["logits"] = y_pred
    #         return output

    #     output["h_classif"] = h_classif
    #     return output

    def extract_embed_impl(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        chunk_length: int = 0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ) -> torch.Tensor:
        """Extract an embedding from the network.

        Args:
            x: Input feature tensor.
            x_lengths: Optional sequence lengths.
            chunk_length: Chunk length for encoder evaluation.
            embed_layer: Head layer index to use.
            detach_chunks: Whether to detach chunk outputs.

        Returns:
            Embedding tensor.
        """
        if embed_layer is None:
            embed_layer = self.embed_layer

        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        if max_in_length <= chunk_length or chunk_length == 0:
            x = self.encoder_net(x, x_lengths=x_lengths)
            if isinstance(x, tuple):
                x = x[0]
        else:
            x = eval_nnet_by_chunks(
                x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
            )

            if x.device != self.device:
                x = x.to(self.device)

        x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
        p = self.pool_net(x, x_lengths=x_lengths)
        if self.proj_head_net is not None:
            return self.proj_head_net(p)

        y = self.classif_net.extract_embed(p, embed_layer)
        return y

    def extract_embed(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        chunk_length: int = 0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ) -> torch.Tensor:
        """Extract embeddings, chunking long utterances when needed.

        Args:
            x: Input feature tensor.
            x_lengths: Optional sequence lengths.
            chunk_length: Chunk length for encoder evaluation.
            embed_layer: Head layer index to use.
            detach_chunks: Whether to detach chunk outputs.

        Returns:
            Embedding tensor.
        """

        if x.size(-1) <= chunk_length or chunk_length == 0:
            return self.extract_embed_impl(x, x_lengths, 0, embed_layer)
        else:
            e = []
            for i in range(x.size(0)):
                x_i = x[i : i + 1]
                if x_lengths is not None:
                    x_i = x_i[..., : int(x_lengths[i])]

                e_i = self.extract_embed_impl(
                    x_i,
                    chunk_length=chunk_length,
                    embed_layer=embed_layer,
                    detach_chunks=detach_chunks,
                )
                e.append(e_i)

            return torch.cat(e, dim=0)

    def extract_embed_slidwin_legacy(
        self,
        x: torch.Tensor,
        win_length: float,
        win_shift: float,
        snip_edges: bool = False,
        feat_frame_length: Optional[float] = None,
        feat_frame_shift: Optional[float] = None,
        chunk_length: int = 0,
        embed_layer: Optional[int] = None,
        detach_chunks: bool = False,
    ) -> torch.Tensor:
        """Extract embeddings on sliding windows using the legacy convention.

        Args:
            x: Input feature tensor.
            win_length: Window length.
            win_shift: Window shift.
            snip_edges: Whether to discard partial windows at the edges.
            feat_frame_length: Feature frame length in milliseconds.
            feat_frame_shift: Feature frame shift in milliseconds.
            chunk_length: Chunk length for encoder evaluation.
            embed_layer: Head layer index to use.
            detach_chunks: Whether to detach chunk outputs.

        Returns:
            Tensor of window embeddings.
        """
        if feat_frame_shift is not None:
            # assume win_length/shift are in secs, transform to frames
            # pass feat times from msecs to secs
            feat_frame_shift = feat_frame_shift / 1000
            feat_frame_length = feat_frame_length / 1000

            # get length and shift in number of feature frames
            win_shift = win_shift / feat_frame_shift  # this can be a float
            win_length = (
                win_length - feat_frame_length + feat_frame_shift
            ) / feat_frame_shift
            assert win_shift > 0.5, "win-length should be longer than feat-frame-length"

        if embed_layer is None:
            embed_layer = self.embed_layer

        max_in_length = x.size(-1)
        x = self._pre_enc(x)
        x = eval_nnet_by_chunks(
            x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
        )

        if x.device != self.device:
            x = x.to(self.device)

        x, _ = self._post_enc(x)
        pin_time = x.size(-1)  # time dim before pooling
        downsample_factor = float(pin_time) / max_in_length
        p = self.pool_net.forward_slidwin(
            x,
            downsample_factor * win_length,
            downsample_factor * win_shift,
            snip_edges=snip_edges,
        )
        # (batch, pool_dim, time)

        p = p.transpose(1, 2).contiguous().view(-1, p.size(1))
        y = (
            self.classif_net.extract_embed(p, embed_layer)
            .view(x.size(0), -1, self.embed_dim)
            .transpose(1, 2)
            .contiguous()
        )

        return y

    def compute_slidwin_timestamps(
        self,
        num_windows: int,
        win_length: float,
        win_shift: float,
        snip_edges: bool = False,
        feat_frame_length: float = 25,
        feat_frame_shift: float = 10,
        feat_snip_edges: bool = False,
    ) -> torch.Tensor:
        """Compute timestamps for sliding-window embeddings.

        Args:
            num_windows: Number of sliding windows.
            win_length: Window length.
            win_shift: Window shift.
            snip_edges: Whether to discard partial windows at the edges.
            feat_frame_length: Feature frame length in milliseconds.
            feat_frame_shift: Feature frame shift in milliseconds.
            feat_snip_edges: Whether feature extraction snips edge frames.

        Returns:
            Timestamp tensor with shape ``(num_windows, 2)``.
        """
        P = self.compute_slidwin_left_padding(
            win_length,
            win_shift,
            snip_edges,
            feat_frame_length,
            feat_frame_shift,
            feat_snip_edges,
        )

        tstamps = (
            torch.as_tensor(
                [
                    [i * win_shift, i * win_shift + win_length]
                    for i in range(num_windows)
                ]
            )
            - P
        )
        tstamps[tstamps < 0] = 0
        return tstamps

    def compute_slidwin_left_padding(
        self,
        win_length: float,
        win_shift: float,
        snip_edges: bool = False,
        feat_frame_length: float = 25,
        feat_frame_shift: float = 10,
        feat_snip_edges: bool = False,
    ) -> float:
        """Compute the left padding used by sliding-window extraction.

        Args:
            win_length: Window length.
            win_shift: Window shift.
            snip_edges: Whether to discard partial windows at the edges.
            feat_frame_length: Feature frame length in milliseconds.
            feat_frame_shift: Feature frame shift in milliseconds.
            feat_snip_edges: Whether feature extraction snips edge frames.

        Returns:
            Left padding in seconds.
        """
        # pass feat times from msecs to secs
        feat_frame_shift = feat_frame_shift / 1000
        feat_frame_length = feat_frame_length / 1000

        # get length and shift in number of feature frames
        H = win_shift / feat_frame_shift
        L = (win_length - feat_frame_length + feat_frame_shift) / feat_frame_shift
        assert L > 0.5, "win-length should be longer than feat-frame-length"

        # compute left padding in case of snip_edges is False
        if snip_edges:
            P1 = 0
        else:
            Q = (
                L - H
            ) / 2  # left padding in frames introduced by x-vector sliding window
            P1 = (
                Q * feat_frame_shift
            )  # left padding in secs introduced by x-vector sliding window

        if feat_snip_edges:
            # left padding introduced when computing acoustic feats
            P2 = 0
        else:
            P2 = (feat_frame_length - feat_frame_shift) / 2

        # total left padding
        return P1 + P2

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration for the model.

        Returns:
            Dictionary with the model configuration.
        """
        enc_cfg = self.encoder_net.get_config()
        pool_cfg = PF.get_config(self.pool_net)
        config = {
            "encoder_cfg": enc_cfg,
            "pool_net": pool_cfg,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_embed_layers": self.num_embed_layers,
            "hid_act": self.hid_act,
            "loss_type": self.loss_type,
            "cos_scale": self.cos_scale,
            "margin": self.margin,
            "margin_warmup_epochs": self.margin_warmup_epochs,
            "intertop_k": self.intertop_k,
            "intertop_margin": self.intertop_margin,
            "num_subcenters": self.num_subcenters,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
            "norm_before": self.norm_before,
            "head_norm_layer": self.head_norm_layer,
            "head_use_norm": self.head_use_norm,
            "head_use_in_norm": self.head_use_in_norm,
            "head_hid_dim": self.head_hid_dim,
            "head_bottleneck_dim": self.head_bottleneck_dim,
            "proj_head_use_norm": self.proj_head_use_norm,
            "proj_head_norm_before": self.proj_head_norm_before,
            "dropout_rate": self.dropout_rate,
            "embed_layer": self.embed_layer,
            "in_feats": self.in_feats,
            "proj_feats": self.proj_feats,
            "head_type": self.head_type,
            "bias_weight_decay": self.bias_weight_decay,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "XVector":
        """Load a model from a config, state dict, or saved file.

        Args:
            file_path: Optional file path to load from.
            cfg: Optional configuration dictionary.
            state_dict: Optional state dictionary.

        Returns:
            Loaded model instance.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        encoder_net = TorchNALoader.load_from_cfg(cfg=cfg["encoder_cfg"])
        del cfg["encoder_cfg"]

        model = cls(encoder_net, **cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    def change_config(
        self,
        override_output: bool = False,
        override_dropouts: bool = False,
        dropout_rate: float = 0,
        num_classes: Optional[int] = None,
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 10,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        head_type: Union[str, XVectorHeadType] = XVectorHeadType.XVECTOR,
    ) -> None:
        """Change the output head or dropout configuration.

        Args:
            override_output: Whether to rebuild the output layer.
            override_dropouts: Whether to override dropout settings.
            dropout_rate: New dropout rate.
            num_classes: New number of classes.
            loss_type: New loss type.
            cos_scale: New scale value for angular-margin losses.
            margin: New margin value.
            margin_warmup_epochs: New margin warmup duration.
            intertop_k: New InterTopK parameter.
            intertop_margin: New InterTopK margin.
            num_subcenters: New number of subcenters.
            head_type: Desired head type.
        """
        logging.info("changing x-vector config")
        if override_output:
            self.rebuild_output_layer(
                num_classes=num_classes,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                head_type=head_type,
            )

        if override_dropouts:
            logging.info("overriding x-vector dropouts")
            self.encoder_net.change_dropouts(dropout_rate)
            self.classif_net.change_dropouts(dropout_rate)

    def rebuild_output_layer(
        self,
        num_classes: Optional[int] = None,
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 10,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        head_type: Union[str, XVectorHeadType] = XVectorHeadType.XVECTOR,
    ) -> None:
        """Rebuild or retune the classification head.

        Args:
            num_classes: Optional new class count.
            loss_type: Loss type for the new head.
            cos_scale: New scale value for angular-margin losses.
            margin: New margin value.
            margin_warmup_epochs: New margin warmup duration.
            intertop_k: New InterTopK parameter.
            intertop_margin: New InterTopK margin.
            num_subcenters: New number of subcenters.
            head_type: Desired head type.
        """
        head_type = XVectorHeadType(head_type)

        if head_type != self.head_type:
            if head_type != XVectorHeadType.XVECTOR:
                raise ValueError(
                    f"unsupported head type transition {self.head_type} -> {head_type}"
                )
            if self.head_type != XVectorHeadType.DINO:
                raise ValueError(
                    f"unsupported head type transition {self.head_type} -> {head_type}"
                )

            logging.info("transforming dino head into x-vector head")
            self.num_embed_layers = 1
            self.head_use_in_norm = (
                self.proj_head_use_norm and self.proj_head_norm_before
            )
            self.head_use_norm = (
                self.proj_head_use_norm and not self.proj_head_norm_before
            )
            self.classif_net = ClassifHead(
                self.proj_head_net.in_feats,
                num_classes,
                embed_dim=self.proj_head_net.out_feats,
                num_embed_layers=1,
                hid_act=None,
                loss_type=loss_type,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
                norm_layer=self.head_norm_layer,
                use_norm=self.proj_head_use_norm,
                norm_before=self.norm_before,
                dropout_rate=self.dropout_rate,
                use_in_norm=self.head_use_in_norm,
            )

            if (
                self.classif_net.fc_blocks[0].linear.bias is not None
                and self.proj_head_net.proj.bias is not None
            ):
                self.classif_net.fc_blocks[0].linear.bias.data.copy_(
                    self.proj_head_net.proj.bias.data
                )

            self.classif_net.fc_blocks[0].linear.weight.data.copy_(
                self.proj_head_net.proj.weight.data
            )
            if self.head_use_norm:
                self.classif_net.fc_blocks[0].bn1.load_state_dict(
                    self.proj_head_net._norm_layer.state_dict()
                )
            del self.proj_head_net
            self.proj_head_net = None
            self.head_type = XVectorHeadType.XVECTOR
            return

        if (
            (self.num_classes is not None and self.num_classes != num_classes)
            or (self.loss_type != loss_type)
            or (
                loss_type == "subcenter-arc-softmax"
                and self.classif_net.num_subcenters != num_subcenters
            )
        ):
            # if we change the number of classes or the loss-type
            # we need to reinitiate the last layer
            logging.info("rebuilding output layer")
            self.classif_net.rebuild_output_layer(
                num_classes,
                loss_type,
                cos_scale,
                margin,
                margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
                num_subcenters=num_subcenters,
            )
            return

        # otherwise we just change the values of s, margin and margin_warmup
        self.classif_net.set_margin(margin)
        self.classif_net.set_margin_warmup_epochs(margin_warmup_epochs)
        self.classif_net.set_cos_scale(cos_scale)
        self.classif_net.set_intertop_k(intertop_k)
        self.classif_net.set_intertop_margin(intertop_margin)
        self.classif_net.set_num_subcenters(num_subcenters)

    def cancel_output_layer_grads(self) -> None:
        """Clear gradients on the classification head output layer."""
        for p in self.classif_net.output.parameters():
            p.grad = None

    def freeze_preembed_layers(self) -> None:
        """Freeze encoder and pooling layers up to the embedding head."""
        self.encoder_net.freeze()
        if self.proj is not None:
            self.proj.freeze()

        for param in self.pool_net.parameters():
            param.requires_grad = False

        layer_list = [l for l in range(self.embed_layer)]
        self.classif_net.freeze_layers(layer_list)

    def set_train_mode(self, mode: str) -> None:
        """Switch between the supported training modes.

        Args:
            mode: Training mode name.
        """
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        elif mode == "ft-embed-affine":
            self.unfreeze()
            self.freeze_preembed_layers()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        if self.head_type == XVectorHeadType.DINO:
            self.classif_net.freeze_output_g()

        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Apply the requested training mode to the module tree.

        Args:
            train_mode: Training mode name.
        """
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode == "ft-embed-affine":
            self.encoder_net.eval()
            if self.proj is not None:
                self.proj.eval()

            self.pool_net.eval()
            self.classif_net.train()
            layer_list = [l for l in range(self.embed_layer)]
            self.classif_net.put_layers_in_eval_mode(layer_list)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Return the prototype affinity matrix.

        Returns:
            Prototype affinity tensor from the classification head.
        """
        return self.classif_net.compute_prototype_affinity()

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Return the supported training modes.

        Returns:
            List of valid training mode names.
        """
        return ["full", "frozen", "ft-embed-affine"]

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter constructor arguments for the base x-vector model.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        # get arguments for pooling
        pool_args = PF.filter_args(**kwargs["pool_net"])
        args = filter_func_args(XVector.__init__, kwargs)
        args["pool_net"] = pool_args
        return args

    @staticmethod
    def add_class_args(
        parser: Any, prefix: Optional[str] = None, skip: set = set()
    ) -> None:
        """Add constructor arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
            skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        PF.add_class_args(
            parser, prefix="pool_net", skip=["dim", "in_feats", "keepdim"]
        )

        parser.add_argument(
            "--head-type",
            default=XVectorHeadType.XVECTOR.value,
            choices=XVectorHeadType.choices(),
            help="type of classification head in [x-vector, dino]",
        )

        parser.add_argument(
            "--embed-dim", default=256, type=int, help=("x-vector dimension")
        )

        parser.add_argument(
            "--num-embed-layers",
            default=1,
            type=int,
            help=("number of layers in the classif head"),
        )

        try:
            parser.add_argument("--hid-act", default="relu", help="hidden activation")
        except:
            pass

        parser.add_argument(
            "--loss-type",
            default="arc-softmax",
            choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
            help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
        )

        parser.add_argument(
            "--cos-scale", default=64, type=float, help="scale for arcface"
        )

        parser.add_argument(
            "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
        )

        parser.add_argument(
            "--margin-warmup-epochs",
            default=10,
            type=float,
            help="number of epoch until we set the final margin",
        )

        parser.add_argument(
            "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
        )
        parser.add_argument(
            "--intertop-margin",
            default=0.0,
            type=float,
            help="margin for InterTopK penalty",
        )

        parser.add_argument(
            "--num-subcenters",
            default=2,
            type=int,
            help="number of subcenters in subcenter losses",
        )

        try:
            parser.add_argument(
                "--norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help="type of normalization layer for all components of x-vector network",
            )
        except:
            pass

        try:
            parser.add_argument(
                "--head-norm-layer",
                default=None,
                choices=[
                    "batch-norm",
                    "group-norm",
                    "instance-norm",
                    "instance-norm-affine",
                    "layer-norm",
                ],
                help=(
                    "type of normalization layer for classification head, "
                    "it overrides the value of the norm-layer parameter"
                ),
            )
        except:
            pass

        parser.add_argument(
            "--use-norm",
            default=True,
            action=ActionYesNo,
            help="without batch normalization",
        )

        parser.add_argument(
            "--norm-before",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton before activation",
        )

        parser.add_argument(
            "--head-use-norm",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton at the head",
        )
        parser.add_argument(
            "--head-use-in-norm",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton at the head input",
        )

        parser.add_argument(
            "--head-hid-dim",
            default=2048,
            type=int,
            help="bottleneck dim of DINO head",
        )

        parser.add_argument(
            "--head-bottleneck-dim",
            default=256,
            type=int,
            help="bottleneck dim of DINO head",
        )

        parser.add_argument(
            "--proj-head-use-norm",
            default=True,
            action=ActionYesNo,
            help="batch normalizaton at projection head",
        )
        parser.add_argument(
            "--proj-head-norm-before",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton at the begining of projection head",
        )

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if "in_feats" not in skip:
            parser.add_argument(
                "--in-feats",
                default=None,
                type=int,
                help=(
                    "input feature dimension, "
                    "if None it will try to infer from encoder network"
                ),
            )

        parser.add_argument(
            "--proj-feats",
            default=None,
            type=int,
            help=(
                "dimension of linear projection after encoder network, "
                "if None, there is not projection"
            ),
        )

        parser.add_argument(
            "--bias-weight-decay",
            default=None,
            type=float,
            help="biases weight decay, if None default it is used",
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
                help="xvector options",
            )

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter arguments for finetuning.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        args = filter_func_args(XVector.change_config, kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add finetuning arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--override-output",
            default=False,
            action=ActionYesNo,
            help="changes the config of the output layer",
        )

        parser.add_argument(
            "--loss-type",
            default="arc-softmax",
            choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
            help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
        )

        parser.add_argument(
            "--cos-scale", default=64, type=float, help="scale for arcface"
        )

        parser.add_argument(
            "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
        )

        parser.add_argument(
            "--margin-warmup-epochs",
            default=10,
            type=float,
            help="number of epoch until we set the final margin",
        )

        parser.add_argument(
            "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
        )
        parser.add_argument(
            "--intertop-margin",
            default=0.0,
            type=float,
            help="margin for InterTopK penalty",
        )

        parser.add_argument(
            "--num-subcenters",
            default=2,
            type=int,
            help="number of subcenters in subcenter losses",
        )

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
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter arguments for DINO teacher configuration.

        Args:
            kwargs: Candidate keyword arguments.

        Returns:
            Filtered configuration dictionary.
        """
        return XVector.filter_finetune_args(**kwargs)

    @staticmethod
    def add_dino_teacher_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Add DINO teacher arguments to an argparse parser.

        Args:
            parser: Parser to extend.
            prefix: Optional prefix for nested parsing.
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
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args
    add_argparse_finetune_args = add_finetune_args

"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HypDataClass
from ....utils.misc import filter_func_args
from ...narchs import (
    HydraClassifHeadOutput,
    HydraHead,
    HydraHeadFactory,
    HydraHeadType,
    HydraRegressionHeadOutput,
    QFormerV2,
    QProjHead,
)
from ...torch_model import TorchModel


@dataclass
class QVectorOutput(HypDataClass):
    """Container for q-vector inference artifacts."""

    qmatrix: torch.Tensor
    """Per-query embeddings returned by the output aggregation Q-former (batch, num_queries, dim)."""

    qvector: torch.Tensor
    """Projected embedding for each input example (batch, qvector_dim)."""

    head_output: Optional[Union[HydraClassifHeadOutput, HydraRegressionHeadOutput]] = (
        None
    )
    """Result produced by the downstream Hydra head (logits/loss or regression output)."""

    backbone_output_feats: Optional[List[torch.Tensor]] = None
    """Optional list of backbone output features that were returned for analysis."""

    backbone_output_feats_lengths: Optional[torch.Tensor] = None
    """Lengths corresponding to `backbone_output_feats` when variable-length inputs are used."""

    backbone_hidden_feats: Optional[List[torch.Tensor]] = None
    """Optional hidden-layer feature maps captured from the backbone encoder."""

    backbone_hidden_feats_lengths: Optional[torch.Tensor] = None
    """Lengths matching `backbone_hidden_feats` for variable-length inputs."""


class QVectorTrainMode(str, Enum):
    """Training modes for the QVector model."""

    FULL = "full"
    FROZEN = "frozen"
    ADAPTERS_QFORMERS = "adapters-qformers"
    QFORMERS = "qformers"
    OUTPUT_FEATS_QFORMER = "output-feats-qformer"
    PROJ_HEAD = "proj-head"
    OUTPUT_LAYER = "output-layer"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of valid training-mode strings."""
        return [o.value for o in QVectorTrainMode]


class QVector(TorchModel):
    """Core implementation of the q-vector encoder/classifier.

    Attributes:
        hidden_feats_queries: Learnable queries attending to hidden backbone layers.
        hidden_feats_agg_qformer: Q-former aggregating hidden-layer features.
        output_feats_queries: Learnable queries attending to output features.
        output_feats_agg_qformer: Q-former aggregating output features.
        proj_head: Projection head that flattens the Q-former output into a vector.
        head: Classification or regression head operating on the q-vector.
        num_hidden_feats_queries: Number of hidden-feature queries.
        num_output_feats_queries: Number of output-feature queries.
        qvector_dim: Dimensionality of the final q-vector embedding.
    """

    def __init__(
        self,
        hidden_feats_agg_qformer: Union[Dict[str, Any], QFormerV2, None],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Union[Dict[str, Any], None],
        num_output_feats_queries: int,
        qvector_dim: int,
        head: Union[Dict[str, Any], HydraHead],
        bias_weight_decay: Optional[float] = None,
    ):
        """Initialize the q-vector model components.

        Args:
            hidden_feats_agg_qformer: Configuration dictionary for the hidden-feature
                aggregation Q-former, or ``None`` to disable hidden aggregation.
            num_hidden_feats_queries: Number of learnable queries applied to hidden
                backbone features.
            output_feats_agg_qformer: Configuration dictionary for the output-feature
                aggregation Q-former. The Q-former is skipped when its ``num_layers`` is
                zero.
            num_output_feats_queries: Number of learnable queries applied to the output
                backbone features.
            qvector_dim: Dimensionality of the projected q-vector embedding.
            head: Keyword arguments used to instantiate the downstream Hydra head.
            bias_weight_decay: Optional weight-decay value applied only to bias
                parameters when building optimizer parameter groups.
        """
        super().__init__(bias_weight_decay=bias_weight_decay)

        assert num_hidden_feats_queries > 0 or num_output_feats_queries > 0
        self.num_hidden_feats_queries = num_hidden_feats_queries
        self.num_output_feats_queries = num_output_feats_queries
        self.qvector_dim = qvector_dim

        query_dim = None
        if num_hidden_feats_queries > 0:
            if isinstance(hidden_feats_agg_qformer, QFormerV2):
                self.hidden_feats_agg_qformer = hidden_feats_agg_qformer
            else:
                logging.info("Building hidden_feats_agg_qformer from config dict")
                hidden_feats_agg_qformer["multilayer_input"] = True
                self.hidden_feats_agg_qformer = QFormerV2(**hidden_feats_agg_qformer)

            query_dim = self.hidden_feats_agg_qformer.hidden_dim
            self.hidden_feats_queries = nn.Parameter(
                torch.zeros((num_hidden_feats_queries, query_dim))
            )
        else:
            self.hidden_feats_queries = None
            self.hidden_feats_agg_qformer = None

        if isinstance(output_feats_agg_qformer, QFormerV2):
            self.output_feats_agg_qformer = output_feats_agg_qformer
        elif isinstance(output_feats_agg_qformer, dict) and (
            output_feats_agg_qformer.get("num_layers", 0) > 0
        ):
            logging.info("Building output_feats_agg_qformer from config dict")
            output_feats_agg_qformer["multilayer_input"] = False
            self.output_feats_agg_qformer = QFormerV2(**output_feats_agg_qformer)
        else:
            self.output_feats_queries = None
            self.output_feats_agg_qformer = None

        if self.output_feats_agg_qformer is None and num_output_feats_queries > 0:
            raise ValueError(
                "num_output_feats_queries > 0 but no output_feats_agg_qformer provided"
            )

        if self.output_feats_agg_qformer is not None:
            if query_dim is not None:
                assert (
                    query_dim == self.output_feats_agg_qformer.hidden_dim
                ), "query_dim mismatch"
            else:
                query_dim = self.output_feats_agg_qformer.hidden_dim

            self.output_feats_queries = nn.Parameter(
                torch.zeros((num_output_feats_queries, query_dim))
            )
            proj_uses_norm = not self.output_feats_agg_qformer.output_is_normalized
            proj_norm_layer = self.output_feats_agg_qformer.norm_layer
            qformer_out_feats = self.output_feats_agg_qformer.out_dim
        else:
            proj_uses_norm = not self.hidden_feats_agg_qformer.output_is_normalized
            proj_norm_layer = self.hidden_feats_agg_qformer.norm_layer
            qformer_out_feats = self.hidden_feats_agg_qformer.out_dim

        qmatrix_dim = (
            num_hidden_feats_queries + num_output_feats_queries
        ) * qformer_out_feats
        logging.info(
            "Building proj_head from qmatrix_dim=%d to qvector_dim=%d uses_norm=%s",
            qmatrix_dim,
            qvector_dim,
            proj_uses_norm,
        )
        self.proj_head = QProjHead(
            in_feats=qmatrix_dim,
            out_feats=qvector_dim,
            use_norm=proj_uses_norm,
            norm_layer=proj_norm_layer,
        )

        if isinstance(head, HydraHead):
            self.head = head
        else:
            logging.info("Building head from config dict")
            head["in_feats"] = qvector_dim
            self.head = HydraHeadFactory.create(**head)

        self._backbone_context = contextlib.nullcontext()
        self._adapter_context = contextlib.nullcontext()
        self._hidden_feats_agg_context = contextlib.nullcontext()
        self._output_feats_agg_context = contextlib.nullcontext()
        self._init_queries()

    def _init_queries(self):
        """Initialise the learnable query tensors using a truncated normal draw."""
        if self.hidden_feats_queries is not None:
            nn.init.trunc_normal_(self.hidden_feats_queries, std=0.02)

        if self.output_feats_queries is not None:
            nn.init.trunc_normal_(self.output_feats_queries, std=0.02)

    @property
    def has_hidden_feats_agg(self):
        return self.hidden_feats_agg_qformer is not None

    @property
    def has_output_feats_agg(self):
        return self.output_feats_agg_qformer is not None

    def _infer_backbone_layers_indices_and_dims(self):
        raise NotImplementedError()

    @property
    def num_classes(self):
        if hasattr(self.head, "num_classes"):
            return self.head.num_classes
        else:
            return None

    @property
    def cos_scale(self):
        if hasattr(self.head, "cos_scale"):
            return self.head.cos_scale
        else:
            return None

    @property
    def margin(self):
        if hasattr(self.head, "margin"):
            return self.head.margin
        else:
            return 0.0

    @property
    def margin_warmup_steps(self):
        if hasattr(self.head, "margin_warmup_steps"):
            return self.head.margin_warmup_steps
        else:
            return 0

    @property
    def intertop_k(self):
        if hasattr(self.head, "intertop_k"):
            return self.head.intertop_k
        else:
            return 0

    @property
    def intertop_margin(self):
        if hasattr(self.head, "intertop_margin"):
            return self.head.intertop_margin
        else:
            return 0.0

    @property
    def num_subcenters(self):
        if hasattr(self.head, "num_subcenters"):
            return self.head.num_subcenters
        else:
            return 0

    @property
    def loss_type(self):
        if hasattr(self.head, "loss_type"):
            return self.head.loss_type
        else:
            raise ValueError("head has no loss_type attribute")

    def update_loss_margin(self, global_step: int):
        """Update margin scheduling for large-margin losses when supported.

        Args:
            global_step: Current optimisation step (or epoch) used to drive the
                scheduler.
        """
        if hasattr(self.head, "update_margin"):
            self.head.update_margin(global_step)

    def update_hyperparams(self, global_step: int):
        """Refresh any head hyperparameters that evolve during training.

        Args:
            global_step: Current optimisation step (or epoch).
        """
        self.update_loss_margin(global_step)

    def init_from_xvector(self, xvector_model: TorchModel):
        """Initialize q-vector model backbone parameters from a pre-trained x-vector model.

        Args:
            xvector_model: Pre-trained x-vector model to use for initialization.
        """
        raise NotImplementedError()

    # def _pre_enc(self, x):
    #     if self.encoder_net.in_dim() == 4 and x.dim() == 3:
    #         x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
    #     return x

    # def _post_enc(self, x, in_lengths=None, max_in_length=None):
    #     if self.encoder_net.out_dim() == 4:
    #         x = x.view(x.size(0), -1, x.size(-1))

    #     if self.proj is not None:
    #         x = self.proj(x)

    #     if in_lengths is not None:
    #         out_lengths = scale_seq_lengths(in_lengths, x.size(-1), max_in_length)
    #     else:
    #         out_lengths = None

    #     return x, out_lengths

    def forward_backbone(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_hidden_feats: bool = False,
    ):
        """Compute backbone features for the provided input signal.

        Args:
            x: Input tensor with shape ``(batch, feats, time)``.
            x_lengths: Optional sequence-length tensor describing valid frames.
            return_hidden_feats: When ``True``, subclasses should also return hidden
                feature maps from intermediate backbone layers.

        Returns:
            Tuple containing the backbone output features, their lengths, the hidden
            feature maps (if requested), and the corresponding lengths.

        Raises:
            NotImplementedError: Subclasses must supply a concrete implementation.
        """
        raise NotImplementedError("forward_backbone_feats not implemented")

    def forward_adapter(
        self,
        backbone_output_feats: torch.Tensor,
        backbone_output_feats_lengths: Optional[torch.Tensor] = None,
        backbone_hidden_feats: Optional[List[torch.Tensor]] = None,
        backbone_hidden_feats_lengths: Optional[List[torch.Tensor]] = None,
    ):
        """Adapt backbone outputs before they are consumed by the Q-formers.

        Args:
            backbone_output_feats: Feature tensor emitted by the backbone front-end.
            backbone_output_feats_lengths: Optional sequence lengths associated with
                ``backbone_output_feats``.
            backbone_hidden_feats: Optional list of hidden feature maps captured inside
                the backbone.
            backbone_hidden_feats_lengths: Optional lengths for each hidden feature
                tensor.

        Returns:
            Tuple of possibly transformed backbone outputs, lengths, hidden features,
            and their lengths. The default implementation is the identity transform.
        """
        return (
            backbone_output_feats,
            backbone_output_feats_lengths,
            backbone_hidden_feats,
            backbone_hidden_feats_lengths,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        return_backbone_feats: bool = False,
        return_head_output: bool = True,
    ):
        """Run a forward pass through the q-vector pipeline.

        Args:
            x: Input tensor with shape ``(batch, feats, time)``.
            x_lengths: Optional sequence-length tensor describing valid frames in
                ``x``.
            target: Optional class labels used when the head computes a loss.
            target_mask: Optional boolean tensor indicating which targets are valid.
            return_backbone_feats: When ``True``, include backbone features in the
                returned payload.
            return_head_output: When ``True``, compute and return the Hydra head
                output (logits/loss or regression predictions).

        Returns:
            QVectorOutput: Structured output containing the q-matrix, q-vector, head
            output, and any requested backbone features.
        """
        with self._backbone_context:
            (
                backbone_output_feats,
                backbone_output_feats_lengths,
                backbone_hidden_feats,
                backbone_hidden_feats_lengths,
            ) = self.forward_backbone(
                audio,
                audio_lengths,
                return_hidden_feats=return_backbone_feats or self.has_hidden_feats_agg,
            )

        with self._adapter_context:
            (
                backbone_output_feats,
                backbone_output_feats_lengths,
                backbone_hidden_feats,
                backbone_hidden_feats_lengths,
            ) = self.forward_adapter(
                backbone_output_feats,
                backbone_output_feats_lengths,
                backbone_hidden_feats,
                backbone_hidden_feats_lengths,
            )

        if self.hidden_feats_agg_qformer is not None:
            assert (
                backbone_hidden_feats is not None
            ), "backbone hidden features are None"
            with self._hidden_feats_agg_context:
                hidden_feats_queries = self.hidden_feats_queries.unsqueeze(0).expand(
                    backbone_hidden_feats.size(0), -1, -1
                )  # (batch, num_queries, dim)
                hidden_feats_agg = self.hidden_feats_agg_qformer(
                    hidden_feats_queries,
                    backbone_hidden_feats,
                    backbone_hidden_feats_lengths,
                )  # (batch, num_queries, dim)
                if self.output_feats_queries is not None:
                    output_feats_queries = torch.cat(
                        (
                            hidden_feats_agg,
                            self.output_feats_queries.unsqueeze(0).expand(
                                hidden_feats_agg.size(0), -1, -1
                            ),
                        ),
                        dim=1,
                    )
                else:
                    qmatrix = hidden_feats_agg

            if not return_backbone_feats:
                backbone_hidden_feats = None
                backbone_hidden_feats_lengths = None
        else:
            output_feats_queries = self.output_feats_queries.unsqueeze(0).expand(
                backbone_output_feats.size(0), -1, -1
            )  # (batch, num_queries, dim)

        if self.output_feats_agg_qformer is not None:
            with self._output_feats_agg_context:
                qmatrix = self.output_feats_agg_qformer(
                    output_feats_queries,
                    backbone_output_feats,
                    backbone_output_feats_lengths,
                )  # (batch, num_queries, dim)

        if not return_backbone_feats:
            backbone_output_feats = None
            backbone_output_feats_lengths = None

        qmatrix_flat = qmatrix.view(qmatrix.size(0), -1)
        qvector = self.proj_head(qmatrix_flat)
        if return_head_output:
            head_output = self.head(qvector, target, target_mask)
        else:
            head_output = None

        output = QVectorOutput(
            qmatrix=qmatrix,
            qvector=qvector,
            head_output=head_output,
            backbone_hidden_feats=(
                backbone_hidden_feats if return_backbone_feats else None
            ),
            backbone_hidden_feats_lengths=(
                backbone_hidden_feats_lengths if return_backbone_feats else None
            ),
            backbone_output_feats=(
                backbone_output_feats if return_backbone_feats else None
            ),
            backbone_output_feats_lengths=(
                backbone_output_feats_lengths if return_backbone_feats else None
            ),
        )
        return output

    # def forward_logits(self, x, x_lengths=None, y=None):
    #     """Forward function

    #     Args:
    #       x: input features tensor with shape=(batch, in_feats, time).
    #       x_lengths: time lengths of the features with shape=(batch,).
    #       y: target classes torch.long tensor with shape=(batch,).

    #     Returns:
    #       class logits tensor with shape=(batch, num_classes).
    #     """
    #     max_in_length = x.size(-1)
    #     x = self._pre_enc(x)
    #     x = self.encoder_net(x)
    #     if isinstance(x, tuple):
    #         x = x[0]

    #     if not torch.all(torch.isfinite(x)):
    #         logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
    #     x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
    #     if not torch.all(torch.isfinite(x)):
    #         logging.warning("non-finite x-enc1-avg=%f", torch.mean(x))
    #     p = self.pool_net(x, x_lengths=x_lengths)
    #     if not torch.all(torch.isfinite(p)):
    #         logging.warning("non-finite p-avg=%f", torch.mean(p))
    #     xvector = None
    #     if self.proj_head_net is not None:
    #         p = self.proj_head_net(p)
    #         xvector = p

    #     logits = self.classif_net(p, y)
    #     if not torch.all(torch.isfinite(logits)):
    #         logging.warning("non-finite y-avg=%f", torch.mean(logits))
    #     # return logits
    #     output = XVectorOutput(None, logits, xvector)
    #     return output

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
    #     else:
    #         y_pred = None

    #     if h_classif is not None:
    #         xvector = h_classif[0]
    #     else:
    #         xvector = None

    #     output = XVectorOutput(None, y_pred, xvector, h_enc, h_classif)
    #     return output

    # def extract_embed_impl(
    #     self, x, x_lengths=None, chunk_length=0, embed_layer=None, detach_chunks=False
    # ):
    #     if embed_layer is None:
    #         embed_layer = self.embed_layer

    #     max_in_length = x.size(-1)
    #     x = self._pre_enc(x)
    #     if max_in_length <= chunk_length or chunk_length == 0:
    #         x = self.encoder_net(x, x_lengths=x_lengths)
    #         if isinstance(x, tuple):
    #             x = x[0]
    #     else:
    #         x = eval_nnet_by_chunks(
    #             x, self.encoder_net, chunk_length, detach_chunks=detach_chunks
    #         )

    #         if x.device != self.device:
    #             x = x.to(self.device)

    #     x, x_lengths = self._post_enc(x, x_lengths, max_in_length)
    #     p = self.pool_net(x, x_lengths=x_lengths)
    #     if self.proj_head_net is not None:
    #         return self.proj_head_net(p)

    #     y = self.classif_net.extract_embed(p, embed_layer)
    #     return y

    # def extract_embed(
    #     self, x, x_lengths=None, chunk_length=0, embed_layer=None, detach_chunks=False
    # ):

    #     if x.size(-1) <= chunk_length or chunk_length == 0:
    #         return self.extract_embed_impl(x, x_lengths, 0, embed_layer)
    #     else:
    #         e = []
    #         for i in range(x.size(0)):
    #             x_i = x[i : i + 1]
    #             if x_lengths is not None:
    #                 x_i = x_i[..., x_lengths[i]]

    #             e_i = self.extract_embed_impl(
    #                 x_i,
    #                 chunk_length=chunk_length,
    #                 embed_layer=embed_layer,
    #                 detach_chunks=detach_chunks,
    #             )
    #             e.append(e_i)

    #         return torch.cat(e, dim=0)

    # def compute_slidwin_timestamps(
    #     self,
    #     num_windows,
    #     win_length,
    #     win_shift,
    #     snip_edges=False,
    #     feat_frame_length=25,
    #     feat_frame_shift=10,
    #     feat_snip_edges=False,
    # ):
    #     P = self.compute_slidwin_left_padding(
    #         win_length,
    #         win_shift,
    #         snip_edges,
    #         feat_frame_length,
    #         feat_frame_shift,
    #         feat_snip_edges,
    #     )

    #     tstamps = (
    #         torch.as_tensor(
    #             [
    #                 [i * win_shift, i * win_shift + win_length]
    #                 for i in range(num_windows)
    #             ]
    #         )
    #         - P
    #     )
    #     tstamps[tstamps < 0] = 0
    #     return tstamps

    # def compute_slidwin_left_padding(
    #     self,
    #     win_length,
    #     win_shift,
    #     snip_edges=False,
    #     feat_frame_length=25,
    #     feat_frame_shift=10,
    #     feat_snip_edges=False,
    # ):
    #     # pass feat times from msecs to secs
    #     feat_frame_shift = feat_frame_shift / 1000
    #     feat_frame_length = feat_frame_length / 1000

    #     # get length and shift in number of feature frames
    #     H = win_shift / feat_frame_shift
    #     L = (win_length - feat_frame_length + feat_frame_shift) / feat_frame_shift
    #     assert L > 0.5, "win-length should be longer than feat-frame-length"

    #     # compute left padding in case of snip_edges is False
    #     if snip_edges:
    #         P1 = 0
    #     else:
    #         Q = (
    #             L - H
    #         ) / 2  # left padding in frames introduced by x-vector sliding window
    #         P1 = (
    #             Q * feat_frame_shift
    #         )  # left padding in secs introduced by x-vector sliding window

    #     if feat_snip_edges:
    #         # left padding introduced when computing acoustic feats
    #         P2 = 0
    #     else:
    #         P2 = (feat_frame_length - feat_frame_shift) / 2

    #     # total left padding
    #     return P1 + P2

    def get_config(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the constructor arguments.

        Returns:
            Dict[str, Any]: Configuration dictionary that can be fed back into the
            constructor (along with subclass-specific backbone parameters).
        """
        head = self.head.get_config()
        head["head_type"] = HydraHeadType.from_instance(self.head)
        config = {
            "hidden_feats_agg_qformer": (
                self.hidden_feats_agg_qformer.get_config()
                if self.hidden_feats_agg_qformer is not None
                else None
            ),
            "num_hidden_feats_queries": self.num_hidden_feats_queries,
            "output_feats_agg_qformer": (
                self.output_feats_agg_qformer.get_config()
                if self.output_feats_agg_qformer is not None
                else None
            ),
            "num_output_feats_queries": self.num_output_feats_queries,
            "qvector_dim": self.qvector_dim,
            "proj_head": self.proj_head.get_config(),
            "head": head,
            "bias_weight_decay": self.bias_weight_decay,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # @classmethod
    # def load(cls, file_path=None, cfg=None, state_dict=None):
    #     cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
    #     encoder_net = TorchNALoader.load_from_cfg(cfg=cfg["encoder_cfg"])
    #     for k in "encoder_cfg":
    #         del cfg[k]

    #     model = cls(encoder_net, **cfg)
    #     if state_dict is not None:
    #         model.load_state_dict(state_dict)

    #     return model

    # def change_config(
    #     self,
    #     override_output=False,
    #     override_dropouts=False,
    #     dropout_rate=0,
    #     num_classes=None,
    #     loss_type="arc-softmax",
    #     cos_scale=64,
    #     margin=0.3,
    #     margin_warmup_epochs=10,
    #     intertop_k=5,
    #     intertop_margin=0.0,
    #     num_subcenters=2,
    #     head_type=XVectorHeadType.XVECTOR,
    # ):
    #     logging.info("changing x-vector config")
    #     if override_output:
    #         self.rebuild_output_layer(
    #             num_classes=num_classes,
    #             loss_type=loss_type,
    #             cos_scale=cos_scale,
    #             margin=margin,
    #             margin_warmup_epochs=margin_warmup_epochs,
    #             intertop_k=intertop_k,
    #             intertop_margin=intertop_margin,
    #             num_subcenters=num_subcenters,
    #             head_type=head_type,
    #         )

    #     if override_dropouts:
    #         logging.info("overriding x-vector dropouts")
    #         self.encoder_net.change_dropouts(dropout_rate)
    #         self.classif_net.change_dropouts(dropout_rate)

    # def rebuild_output_layer(
    #     self,
    #     num_classes=None,
    #     loss_type="arc-softmax",
    #     cos_scale=64,
    #     margin=0.3,
    #     margin_warmup_epochs=10,
    #     intertop_k=5,
    #     intertop_margin=0.0,
    #     num_subcenters=2,
    #     head_type=XVectorHeadType.XVECTOR,
    # ):

    #     if head_type != self.head_type:
    #         # only from dino to x-vector
    #         assert self.head_type == XVectorHeadType.DINO
    #         logging.info("transforming dino head into x-vector head")
    #         self.num_embed_layers = 1
    #         self.head_use_in_norm = (
    #             self.proj_head_use_norm and self.proj_head_norm_before
    #         )
    #         self.head_use_norm = (
    #             self.proj_head_use_norm and not self.proj_head_norm_before
    #         )
    #         self.classif_net = ClassifHead(
    #             self.proj_head_net.in_feats,
    #             num_classes,
    #             embed_dim=self.proj_head_net.out_feats,
    #             num_embed_layers=1,
    #             hid_act=None,
    #             loss_type=loss_type,
    #             cos_scale=cos_scale,
    #             margin=margin,
    #             margin_warmup_epochs=margin_warmup_epochs,
    #             intertop_k=intertop_k,
    #             intertop_margin=intertop_margin,
    #             num_subcenters=num_subcenters,
    #             norm_layer=self.head_norm_layer,
    #             use_norm=self.proj_head_use_norm,
    #             norm_before=self.norm_before,
    #             dropout_rate=self.dropout_rate,
    #             use_in_norm=self.head_use_in_norm,
    #         )

    #         if (
    #             self.classif_net.fc_blocks[0].linear.bias is not None
    #             and self.proj_head_net.proj.bias is not None
    #         ):
    #             self.classif_net.fc_blocks[0].linear.bias.data.copy_(
    #                 self.proj_head_net.proj.bias.data
    #             )

    #         self.classif_net.fc_blocks[0].linear.weight.data.copy_(
    #             self.proj_head_net.proj.weight.data
    #         )
    #         if self.head_use_norm:
    #             self.classif_net.fc_blocks[0].bn1.load_state_dict(
    #                 self.proj_head_net._norm_layer.state_dict()
    #             )
    #         del self.proj_head_net
    #         self.proj_head_net = None
    #         self.head_type = XVectorHeadType.XVECTOR
    #         return

    #     if (
    #         (self.num_classes is not None and self.num_classes != num_classes)
    #         or (self.loss_type != loss_type)
    #         or (
    #             loss_type == "subcenter-arc-softmax"
    #             and self.classif_net.num_subcenters != num_subcenters
    #         )
    #     ):
    #         # if we change the number of classes or the loss-type
    #         # we need to reinitiate the last layer
    #         logging.info("rebuilding output layer")
    #         self.classif_net.rebuild_output_layer(
    #             num_classes,
    #             loss_type,
    #             cos_scale,
    #             margin,
    #             margin_warmup_epochs,
    #             intertop_k=intertop_k,
    #             intertop_margin=intertop_margin,
    #             num_subcenters=num_subcenters,
    #         )
    #         return

    #     # otherwise we just change the values of s, margin and margin_warmup
    #     self.classif_net.set_margin(margin)
    #     self.classif_net.set_margin_warmup_epochs(margin_warmup_epochs)
    #     self.classif_net.set_cos_scale(cos_scale)
    #     self.classif_net.set_intertop_k(intertop_k)
    #     self.classif_net.set_intertop_margin(intertop_margin)
    #     self.classif_net.set_num_subcenters(num_subcenters)

    # def cancel_output_layer_grads(self):
    #     for p in self.classif_net.output.parameters():
    #         p.grad = None

    def set_train_mode(self, mode: Union[str, QVectorTrainMode]):
        """Switch between predefined training regimes.

        Args:
            mode: Target training mode as a string or ``QVectorTrainMode`` enum.

        Raises:
            ValueError: If the requested mode is not supported.
        """
        if mode == self._train_mode:
            return

        self._backbone_context = contextlib.nullcontext()
        self._adapter_context = contextlib.nullcontext()
        self._hidden_feats_agg_context = contextlib.nullcontext()
        self._output_feats_agg_context = contextlib.nullcontext()

        if mode == QVectorTrainMode.FULL:
            self.unfreeze()
        elif mode == QVectorTrainMode.FROZEN:
            self.freeze()
        elif mode == QVectorTrainMode.ADAPTERS_QFORMERS:
            self._backbone_context = torch.no_grad()
            self.unfreeze()
            self.freeze_backbone()
        elif mode == QVectorTrainMode.QFORMERS:
            self._backbone_context = torch.no_grad()
            self._adapter_context = torch.no_grad()
            self.unfreeze()
            self.freeze_backbone()
            self.freeze_adapters()
        elif mode == QVectorTrainMode.OUTPUT_FEATS_QFORMER:
            self._backbone_context = torch.no_grad()
            self._adapter_context = torch.no_grad()
            self._hidden_feats_agg_context = torch.no_grad()
            self.unfreeze()
            self.freeze_backbone()
            self.freeze_adapters()
            self.hidden_feats_agg_qformer.freeze()
        elif mode == QVectorTrainMode.PROJ_HEAD:
            self._backbone_context = torch.no_grad()
            self._adapter_context = torch.no_grad()
            self._hidden_feats_agg_context = torch.no_grad()
            self._output_feats_agg_context = torch.no_grad()
            self.freeze()
            self.proj_head.unfreeze()
            self.head.unfreeze()
        elif mode == QVectorTrainMode.OUTPUT_LAYER:
            self._backbone_context = torch.no_grad()
            self._adapter_context = torch.no_grad()
            self._hidden_feats_agg_context = torch.no_grad()
            self._output_feats_agg_context = torch.no_grad()
            self.freeze()
            self.head.unfreeze()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        self._train_mode = mode

    def _train(self, train_mode: Union[str, QVectorTrainMode]):
        """Override ``nn.Module.train`` to honour custom training regimes.

        Args:
            train_mode: Target training mode.

        Raises:
            ValueError: If the training mode is unknown.
        """
        if train_mode in [QVectorTrainMode.FULL, QVectorTrainMode.FROZEN]:
            super()._train(str(train_mode))
        elif train_mode == QVectorTrainMode.ADAPTERS_QFORMERS:
            self.set_backbone_in_eval_mode()
            self.set_adapters_in_train_mode()
            self.hidden_feats_agg_qformer.train()
            self.output_feats_agg_qformer.train()
            self.proj_head.train()
            self.head.train()
        elif train_mode == QVectorTrainMode.QFORMERS:
            self.set_backbone_in_eval_mode()
            self.set_adapters_in_eval_mode()
            self.hidden_feats_agg_qformer.train()
            self.output_feats_agg_qformer.train()
            self.proj_head.train()
            self.head.train()
        elif train_mode == QVectorTrainMode.OUTPUT_FEATS_QFORMER:
            self.set_backbone_in_eval_mode()
            self.set_adapters_in_eval_mode()
            self.hidden_feats_agg_qformer.eval()
            self.output_feats_agg_qformer.train()
            self.proj_head.train()
            self.head.train()
        elif train_mode == QVectorTrainMode.PROJ_HEAD:
            self.set_backbone_in_eval_mode()
            self.set_adapters_in_eval_mode()
            self.hidden_feats_agg_qformer.eval()
            self.output_feats_agg_qformer.eval()
            self.proj_head.train()
            self.head.train()
        elif train_mode == QVectorTrainMode.OUTPUT_LAYER:
            self.set_backbone_in_eval_mode()
            self.set_adapters_in_eval_mode()
            self.hidden_feats_agg_qformer.eval()
            self.output_feats_agg_qformer.eval()
            self.proj_head.eval()
            self.head.train()
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    def freeze_backbone(self):
        """Freeze backbone parameters. Subclasses must implement the details."""
        raise NotImplementedError("set_freeze_backbone not implemented")

    def freeze_adapters(self):
        """Freeze adapter modules. Subclasses must implement the details."""
        raise NotImplementedError("set_freeze_adapters not implemented")

    def set_backbone_in_eval_mode(self):
        """Put the backbone into evaluation mode. Subclasses decide the specifics."""
        raise NotImplementedError("set_backbone_in_eval_mode not implemented")

    def set_adapters_in_train_mode(self):
        """Put adapter modules into training mode. Subclasses must implement it."""
        raise NotImplementedError("set_adapters_in_train_mode not implemented")

    def set_adapters_in_eval_mode(self):
        """Put adapter modules into evaluation mode. Subclasses must implement it."""
        raise NotImplementedError("set_adapters_in_eval_mode not implemented")

    def compute_prototype_affinity(self):
        """Return prototype affinity matrix when the head exposes it.

        Returns:
            torch.Tensor: Affinity matrix measuring cosine similarity between class
            prototypes.

        Raises:
            NotImplementedError: If the active head does not implement prototype
                affinity computation.
        """
        if hasattr(self.head, "compute_prototype_affinity"):
            return self.head.compute_prototype_affinity()
        else:
            raise NotImplementedError(
                "compute_prototype_affinity is not implemented for this head type"
            )

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(QVector.__init__)
        raise NotImplementedError()

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI/configuration arguments for QVector models.

        Args:
            parser: Target parser where the options will be registered.
            prefix: Optional namespace prefix for the registered arguments.
            skip: Optional set of argument names that should be omitted.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        skip = set(skip) if skip is not None else set()

        if "num_hidden_feats_queries" not in skip:
            parser.add_argument(
                "--num-hidden-feats-queries",
                type=int,
                default=0,
                help="number of learned queries attending to hidden backbone features",
            )

        if "num_output_feats_queries" not in skip:
            parser.add_argument(
                "--num-output-feats-queries",
                type=int,
                default=0,
                help="number of learned queries attending to output backbone features",
            )

        if "qvector_dim" not in skip:
            parser.add_argument(
                "--qvector-dim",
                type=int,
                default=256,
                help="final q-vector embedding dimension",
            )

        if "bias_weight_decay" not in skip:
            parser.add_argument(
                "--bias-weight-decay",
                type=float,
                default=None,
                help="optional bias-only weight decay value",
            )

        if "hidden_feats_agg_qformer" not in skip:
            hidden_skip = {"multilayer_input"}
            hidden_skip.update(skip)
            QFormerV2.add_class_args(
                parser,
                prefix="hidden_feats_agg_qformer",
                skip=hidden_skip,
            )

        if "output_feats_agg_qformer" not in skip:
            output_skip = {"multilayer_input"}
            output_skip.update(skip)
            QFormerV2.add_class_args(
                parser,
                prefix="output_feats_agg_qformer",
                skip=output_skip,
            )

        if "head" not in skip:
            HydraHeadFactory.add_class_args(
                parser,
                prefix="head",
                skip=skip,
            )

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    # @staticmethod
    # def filter_finetune_args(**kwargs):
    #     args = filter_func_args(QVector.change_config, kwargs)
    #     return args

    # @staticmethod
    # def add_finetune_args(parser, prefix=None):
    #     if prefix is not None:
    #         outer_parser = parser
    #         parser = ArgumentParser(prog="")

    #     parser.add_argument(
    #         "--override-output",
    #         default=False,
    #         action=ActionYesNo,
    #         help="changes the config of the output layer",
    #     )

    #     parser.add_argument(
    #         "--loss-type",
    #         default="arc-softmax",
    #         choices=["softmax", "arc-softmax", "cos-softmax", "subcenter-arc-softmax"],
    #         help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
    #     )

    #     parser.add_argument(
    #         "--cos-scale", default=64, type=float, help="scale for arcface"
    #     )

    #     parser.add_argument(
    #         "--margin", default=0.3, type=float, help="margin for arcface, cosface,..."
    #     )

    #     parser.add_argument(
    #         "--margin-warmup-epochs",
    #         default=10,
    #         type=float,
    #         help="number of epoch until we set the final margin",
    #     )

    #     parser.add_argument(
    #         "--intertop-k", default=5, type=int, help="K for InterTopK penalty"
    #     )
    #     parser.add_argument(
    #         "--intertop-margin",
    #         default=0.0,
    #         type=float,
    #         help="margin for InterTopK penalty",
    #     )

    #     parser.add_argument(
    #         "--num-subcenters",
    #         default=2,
    #         type=int,
    #         help="number of subcenters in subcenter losses",
    #     )

    #     try:
    #         parser.add_argument(
    #             "--override-dropouts",
    #             default=False,
    #             action=ActionYesNo,
    #             help=(
    #                 "whether to use the dropout probabilities passed in the "
    #                 "arguments instead of the defaults in the pretrained model."
    #             ),
    #         )
    #     except:
    #         pass

    #     try:
    #         parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
    #     except:
    #         pass

    #     if prefix is not None:
    #         outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

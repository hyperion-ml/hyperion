"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils import HyperDataClass
from ....utils.misc import filter_func_args
from ...hyper_torch_model import HyperTorchModel
from ...layers import GlobalPool1dFactory as PF
from ...losses.sig_reg import SIGReg
from ...narchs import (
    HydraClassifHeadOutput,
    HydraHead,
    HydraHeadFactory,
    HydraRegressionHeadOutput,
    ProjHead,
)


@dataclass
class XVectorPOutput(HyperDataClass):
    """Container for x-vector+ inference artifacts.

    Attributes:
        xvector: Projected embedding for each input example.
        xvector_sig_reg: Optional SIGReg statistic for the embeddings.
        head_output: Optional downstream Hydra head output.
        backbone_output_feats: Optional tensor of backbone output features.
        backbone_output_feats_lengths: Optional lengths for
            ``backbone_output_feats``.
        backbone_hidden_feats: Optional list of backbone hidden features.
        backbone_hidden_feats_lengths: Optional lengths for
            ``backbone_hidden_feats``.
    """

    xvector: torch.Tensor
    """Projected embedding for each input example (batch, xvector_dim)."""

    head_output: Optional[Union[HydraClassifHeadOutput, HydraRegressionHeadOutput]] = (
        None
    )
    """Result produced by the downstream Hydra head (logits/loss or regression output)."""

    backbone_output_feats: Optional[torch.Tensor] = None
    """Optional tensor of backbone output features that were returned for analysis."""

    backbone_output_feats_lengths: Optional[torch.Tensor] = None
    """Lengths corresponding to `backbone_output_feats` when variable-length inputs are used."""

    backbone_hidden_feats: Optional[List[torch.Tensor]] = None
    """Optional hidden-layer feature maps captured from the backbone encoder."""

    backbone_hidden_feats_lengths: Optional[Union[List[torch.Tensor], torch.Tensor]] = (
        None
    )
    """Lengths matching `backbone_hidden_feats` for variable-length inputs."""

    xvector_sig_reg: Optional[torch.Tensor] = None
    """Optional SIGReg statistic for the returned embeddings."""

    @classmethod
    def concatenate(cls, outputs: List["XVectorPOutput"]) -> "XVectorPOutput":
        """Concatenate multiple outputs along the batch dimension.

        Args:
            outputs: Sequence of chunk-level outputs to concatenate.

        Returns:
            XVectorPOutput: Single output covering the concatenated batch.
        """
        if not outputs:
            raise ValueError("Cannot concatenate an empty list of XVectorPOutput.")

        def _cat_optional_tensor(attr: str) -> Optional[torch.Tensor]:
            tensors = [getattr(out, attr) for out in outputs]
            if any(t is None for t in tensors):
                return None
            return torch.cat(tensors, dim=0)

        def _cat_optional_tensor_lists(attr: str) -> Optional[List[torch.Tensor]]:
            tensor_lists = [getattr(out, attr) for out in outputs]
            if any(t_list is None for t_list in tensor_lists):
                return None
            num_entries = len(tensor_lists[0])
            if any(len(t_list) != num_entries for t_list in tensor_lists):
                raise ValueError(
                    f"Inconsistent {attr} across outputs: list lengths differ."
                )
            concatenated: List[torch.Tensor] = []
            for idx in range(num_entries):
                concatenated.append(
                    torch.cat([t_list[idx] for t_list in tensor_lists], dim=0)
                )
            return concatenated

        def _cat_optional_tensor_or_tensor_lists(
            attr: str,
        ) -> Optional[Union[List[torch.Tensor], torch.Tensor]]:
            tensors_or_lists = [getattr(out, attr) for out in outputs]
            if any(t is None for t in tensors_or_lists):
                return None

            if all(isinstance(t, torch.Tensor) for t in tensors_or_lists):
                return torch.cat(tensors_or_lists, dim=0)

            if not all(isinstance(t, list) for t in tensors_or_lists):
                raise ValueError(
                    f"Inconsistent {attr} across outputs: mixed tensor and list values."
                )

            num_entries = len(tensors_or_lists[0])
            if any(len(t_list) != num_entries for t_list in tensors_or_lists):
                raise ValueError(
                    f"Inconsistent {attr} across outputs: list lengths differ."
                )

            concatenated: List[torch.Tensor] = []
            for idx in range(num_entries):
                concatenated.append(
                    torch.cat([t_list[idx] for t_list in tensors_or_lists], dim=0)
                )
            return concatenated

        head_outputs_raw = [out.head_output for out in outputs]
        num_none = sum(h is None for h in head_outputs_raw)
        if 0 < num_none < len(head_outputs_raw):
            raise ValueError(
                "Inconsistent head_output across outputs: mixed None and non-None values."
            )

        head_outputs = [h for h in head_outputs_raw if h is not None]
        head_output: Optional[
            Union[HydraClassifHeadOutput, HydraRegressionHeadOutput]
        ] = None
        if head_outputs:
            first_output = head_outputs[0]
            if not all(isinstance(h, type(first_output)) for h in head_outputs):
                raise ValueError(
                    "All head outputs must share the same type to concatenate."
                )
            if isinstance(first_output, HydraClassifHeadOutput):
                logits = torch.cat([h.logits for h in head_outputs], dim=0)
                weights = torch.tensor(
                    [out.xvector.size(0) for out in outputs],
                    device=logits.device,
                    dtype=logits.dtype,
                )
                loss = None
                if all(h.loss is not None for h in head_outputs):
                    losses = torch.stack([h.loss for h in head_outputs])
                    loss = torch.sum(losses * weights) / torch.sum(weights)
                prototype_code_rate = None
                if all(h.prototype_code_rate is not None for h in head_outputs):
                    prototype_rates = torch.stack(
                        [h.prototype_code_rate for h in head_outputs]
                    )
                    prototype_code_rate = torch.sum(
                        prototype_rates * weights
                    ) / torch.sum(weights)
                prototype_sig_reg = None
                if all(h.prototype_sig_reg is not None for h in head_outputs):
                    prototype_sig_reg = torch.sum(
                        torch.stack([h.prototype_sig_reg for h in head_outputs])
                        * weights
                    ) / torch.sum(weights)
                head_output = HydraClassifHeadOutput(
                    logits=logits,
                    prototype_sig_reg=prototype_sig_reg,
                    loss=loss,
                    prototype_code_rate=prototype_code_rate,
                )
            elif isinstance(first_output, HydraRegressionHeadOutput):
                preds = torch.cat([h.preds for h in head_outputs], dim=0)
                loss = None
                if all(h.loss is not None for h in head_outputs):
                    weights = torch.tensor(
                        [out.xvector.size(0) for out in outputs],
                        device=preds.device,
                        dtype=preds.dtype,
                    )
                    losses = torch.stack([h.loss for h in head_outputs])
                    loss = torch.sum(losses * weights) / torch.sum(weights)
                head_output = HydraRegressionHeadOutput(preds=preds, loss=loss)

        return cls(
            xvector=torch.cat([out.xvector for out in outputs], dim=0),
            head_output=head_output,
            backbone_output_feats=_cat_optional_tensor("backbone_output_feats"),
            backbone_output_feats_lengths=_cat_optional_tensor(
                "backbone_output_feats_lengths"
            ),
            backbone_hidden_feats=_cat_optional_tensor_lists("backbone_hidden_feats"),
            backbone_hidden_feats_lengths=_cat_optional_tensor_or_tensor_lists(
                "backbone_hidden_feats_lengths"
            ),
        )

    @classmethod
    def weighted_average_by_index(
        cls,
        concatenated_output: "XVectorPOutput",
        audio_index: torch.Tensor,
        chunk_weights: Optional[torch.Tensor] = None,
    ) -> "XVectorPOutput":
        """Aggregate chunk-level outputs into per-example averages.

        Args:
            concatenated_output: Concatenated chunk-level output.
            audio_index: Tensor mapping each chunk to its source example.
            chunk_weights: Optional per-chunk weights. When omitted, each chunk
                contributes equally.

        Returns:
            XVectorPOutput: Output averaged back to the original example level.
        """
        if audio_index.ndim != 1:
            raise ValueError("audio_index must be one-dimensional")
        if concatenated_output.xvector.size(0) != audio_index.size(0):
            raise ValueError(
                "audio_index length must match the number of chunk outputs."
            )

        device = concatenated_output.xvector.device
        dtype = concatenated_output.xvector.dtype
        audio_index = audio_index.to(device=device, dtype=torch.long)

        if chunk_weights is None:
            chunk_weights = torch.ones(audio_index.size(0), device=device, dtype=dtype)
        else:
            chunk_weights = chunk_weights.to(device=device, dtype=dtype)

        if audio_index.numel() == 0:
            raise ValueError("audio_index must have at least one element.")

        if torch.any(audio_index < 0):
            raise ValueError("audio_index values must be nonnegative")
        if chunk_weights.shape != audio_index.shape or not torch.all(
            torch.isfinite(chunk_weights) & (chunk_weights >= 0)
        ):
            raise ValueError(
                "chunk_weights must contain a finite nonnegative weight per chunk"
            )
        num_examples = int(audio_index.max().item()) + 1
        weight_sums = torch.zeros(num_examples, device=device, dtype=dtype)
        weight_sums.index_add_(0, audio_index, chunk_weights)
        if not torch.all(weight_sums > 0):
            raise ValueError("Weights must sum to a positive value for every example")

        weighted_xvectors = concatenated_output.xvector * chunk_weights.unsqueeze(1)
        xvector = torch.zeros(
            (num_examples, concatenated_output.xvector.size(1)),
            device=device,
            dtype=dtype,
        )
        xvector.index_add_(0, audio_index, weighted_xvectors)
        xvector = xvector / weight_sums.unsqueeze(1)

        input_head_output = concatenated_output.head_output
        aggregated_head_output: Optional[
            Union[HydraClassifHeadOutput, HydraRegressionHeadOutput]
        ]
        if input_head_output is None:
            aggregated_head_output = None
        elif isinstance(input_head_output, HydraClassifHeadOutput):
            logits_chunks = input_head_output.logits
            weighted_logits = logits_chunks * chunk_weights.unsqueeze(1)
            logits = torch.zeros(
                (num_examples, logits_chunks.size(1)),
                device=logits_chunks.device,
                dtype=logits_chunks.dtype,
            )
            logits.index_add_(0, audio_index, weighted_logits)
            logits = logits / weight_sums.unsqueeze(1)
            aggregated_head_output = HydraClassifHeadOutput(
                logits=logits,
                loss=input_head_output.loss,
                prototype_code_rate=input_head_output.prototype_code_rate,
                prototype_sig_reg=input_head_output.prototype_sig_reg,
            )
        elif isinstance(input_head_output, HydraRegressionHeadOutput):
            preds_chunks = input_head_output.preds
            expand_shape = (preds_chunks.size(0),) + (1,) * (preds_chunks.dim() - 1)
            weighted_preds = preds_chunks * chunk_weights.view(expand_shape)
            preds = torch.zeros(
                (num_examples,) + preds_chunks.shape[1:],
                device=preds_chunks.device,
                dtype=preds_chunks.dtype,
            )
            preds.index_add_(0, audio_index, weighted_preds)
            view_shape = (num_examples,) + (1,) * (preds_chunks.dim() - 1)
            preds = preds / weight_sums.view(view_shape)
            aggregated_head_output = HydraRegressionHeadOutput(
                preds=preds, loss=input_head_output.loss
            )
        else:
            aggregated_head_output = None

        return cls(
            xvector=xvector,
            head_output=aggregated_head_output,
            backbone_output_feats=None,
            backbone_output_feats_lengths=None,
            backbone_hidden_feats=None,
            backbone_hidden_feats_lengths=None,
        )


class XVectorPTrainMode(str, Enum):
    """Training modes for the XVectorP model."""

    FULL = "full"
    FROZEN = "frozen"
    FROZEN_FEAT_EXTRACTOR = "frozen-feat-extractor"
    POOLING = "pooling"
    PROJ_HEAD = "proj-head"
    OUTPUT_LAYER = "output-layer"

    @staticmethod
    def choices() -> List[str]:
        """Return the list of valid training-mode strings.

        Returns:
            List of accepted training-mode values.
        """
        return [o.value for o in XVectorPTrainMode]


class XVectorP(HyperTorchModel):
    """Base x-vector plus (x-vector+) model using pooling and projection.

    Subclasses provide backbone features with shape ``(batch, time, features)``
    and report their feature dimension through ``backbone_output_feats``.

    Attributes:
        pooling: Global pooling module operating over backbone frames.
        proj_head: Projection with optional normalization before or after it.
        head: Optional classification or regression head operating on the embedding.
        xvector_dim: Dimensionality of the projected embedding.
        proj_use_norm: Whether projection normalization is enabled.
        proj_norm_layer: Normalization type; ``None`` selects batch normalization when enabled.
        proj_norm_before: Whether normalization precedes the projection.
        pooling_weight_decay: Optional pooling weight decay.
        proj_weight_decay: Optional projection weight decay.
        head_weight_decay: Optional downstream head weight decay.
        bias_weight_decay: Optional bias and normalization weight decay (inherited).
        max_input_length: Buffer recording the longest training input in samples.
        enable_xvector_sig_reg: Whether projected embeddings are regularized.
        xvector_sig_reg: Optional SIGReg module using global statistics by default.
        train_mode: Active training regime (inherited).
    """

    def __init__(
        self,
        pooling: Union[str, Dict[str, Any], nn.Module],
        xvector_dim: int,
        head: Optional[Union[Dict[str, Any], HydraHead]],
        proj_norm_layer: Optional[str] = None,
        proj_use_norm: bool = True,
        proj_norm_before: bool = True,
        enable_xvector_sig_reg: bool = False,
        xvector_sig_reg: Optional[Dict[str, Any]] = None,
        pooling_weight_decay: Optional[float] = None,
        proj_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialize the x-vector+ model components.

        Args:
            pooling: Configuration for the pooling layer.
            xvector_dim: Dimensionality of the projected x-vector embedding.
            head: Hydra head configuration or module, or ``None`` for an embedding-only model.
            proj_norm_layer: Batch, layer, or RMS normalization; ``None`` uses batch norm when enabled.
            proj_use_norm: Whether to normalize the projection input or output.
            proj_norm_before: Whether normalization precedes projection. Projection
                bias is disabled when normalization follows projection.
            enable_xvector_sig_reg: Whether to calculate SIGReg on projected embeddings.
            xvector_sig_reg: SIGReg constructor arguments; defaults to global statistics.
            pooling_weight_decay: Optional weight-decay override for pooling parameters.
            proj_weight_decay: Optional weight-decay override applied to
                projection-head parameters.
            head_weight_decay: Optional weight-decay override applied to downstream
                head parameters.
            bias_weight_decay: Optional weight decay for biases and normalization
                parameters when building optimizer parameter groups.
        """
        super().__init__(bias_weight_decay=bias_weight_decay)
        self.pooling_weight_decay = pooling_weight_decay
        self.proj_weight_decay = proj_weight_decay
        self.head_weight_decay = head_weight_decay

        self.enable_xvector_sig_reg = enable_xvector_sig_reg
        self.xvector_sig_reg_args = dict(xvector_sig_reg or {})
        if enable_xvector_sig_reg:
            sig_reg_args = dict(self.xvector_sig_reg_args)
            sig_reg_args["distributed_mode"] = "global_data"
            self.xvector_sig_reg = SIGReg(**sig_reg_args)

        self.xvector_dim = xvector_dim
        self.proj_use_norm = proj_use_norm
        self.proj_norm_layer = proj_norm_layer
        self.proj_norm_before = proj_norm_before

        encoder_feats = self.backbone_output_feats()

        self.pooling = self._make_pooling(pooling, encoder_feats)
        pooling_feats = int(encoder_feats * self.pooling.size_multiplier)

        logging.info(
            "Building proj_head from pooling_dim=%d to xvector_dim=%d uses_norm=%s",
            pooling_feats,
            xvector_dim,
            self.proj_use_norm,
        )
        self.proj_head = ProjHead(
            in_feats=pooling_feats,
            out_feats=xvector_dim,
            norm_layer=self.proj_norm_layer,
            use_norm=self.proj_use_norm,
            norm_before=self.proj_norm_before,
        )

        if isinstance(head, HydraHead):
            self.head: Optional[HydraHead] = head
        elif head is None:
            self.head = None
        else:
            logging.info("Building head from config dict")
            self.head = HydraHeadFactory.create(**{**head, "in_feats": xvector_dim})

        self._backbone_context = contextlib.nullcontext()
        self._adapter_context = contextlib.nullcontext()
        self._pooling_context = contextlib.nullcontext()
        self._proj_context = contextlib.nullcontext()
        self.register_buffer("max_input_length", torch.tensor(0, dtype=torch.long))

    def _make_pooling(
        self,
        pooling: Union[str, Dict[str, Any], nn.Module],
        enc_feats: int,
    ) -> nn.Module:
        """Build the global pooling block.

        Args:
            pooling: Pooling configuration string, dictionary, or module.
            enc_feats: Input feature dimension from the encoder.

        Returns:
            Pooling module.
        """
        if isinstance(pooling, str):
            pooling = {"pool_type": pooling}

        if isinstance(pooling, dict):
            pooling = PF.create(**{**pooling, "in_feats": enc_feats})
        if not isinstance(pooling, nn.Module):
            raise TypeError(
                "pooling must be a type string, configuration dictionary, or module"
            )
        if getattr(pooling, "dim", -1) not in (-1, 2) or getattr(
            pooling, "keepdim", False
        ):
            raise ValueError(
                "pooling must reduce the time dimension without keeping it"
            )
        return pooling

    @property
    def max_chunk_length(self) -> int:
        """Maximum chunk length (in samples) seen during training.

        Returns:
            Current maximum chunk length, measured in samples.
        """
        return int(self.max_input_length.item())

    @property
    def num_classes(self) -> Optional[int]:
        """Return the number of classes exposed by the head, if any.

        Returns:
            Number of classes, or ``None`` when unavailable.
        """
        if hasattr(self.head, "num_classes"):
            return self.head.num_classes
        else:
            return None

    @property
    def requires_max_train_length(self) -> bool:
        """Whether training requires a maximum chunk length.

        Returns:
            ``False`` for the base implementation.
        """
        return False

    @property
    def sample_frequency(self) -> float:
        """Return the sample rate expected by the model.

        Returns:
            Sampling frequency in hertz.
        """
        raise NotImplementedError()

    def backbone_output_feats(self) -> int:
        """Return the feature dimension emitted by the backbone or adapter.

        Returns:
            Number of features per frame consumed by global pooling.
        """
        raise NotImplementedError("backbone_output_feats is not implemented")

    @property
    def cos_scale(self) -> Optional[float]:
        """Return the angular-margin scale used by the head, if exposed.

        Returns:
            Cosine scaling factor, or ``None`` when unavailable.
        """
        if hasattr(self.head, "cos_scale"):
            return self.head.cos_scale
        else:
            return None

    @property
    def margin(self) -> float:
        """Return the current margin used by the head.

        Returns:
            Current margin value.
        """
        if hasattr(self.head, "margin"):
            return self.head.margin
        else:
            return 0.0

    @property
    def margin_warmup_steps(self) -> int:
        """Return the margin warmup schedule length, if exposed.

        Returns:
            Number of warmup steps, or ``0`` when unavailable.
        """
        if hasattr(self.head, "margin_warmup_steps"):
            return self.head.margin_warmup_steps
        else:
            return 0

    @property
    def intertop_k(self) -> int:
        """Return the InterTopK `k` value, if exposed.

        Returns:
            InterTopK `k`, or ``0`` when unavailable.
        """
        if hasattr(self.head, "intertop_k"):
            return self.head.intertop_k
        else:
            return 0

    @property
    def intertop_margin(self) -> float:
        """Return the InterTopK margin, if exposed.

        Returns:
            InterTopK margin, or ``0.0`` when unavailable.
        """
        if hasattr(self.head, "intertop_margin"):
            return self.head.intertop_margin
        else:
            return 0.0

    @property
    def num_subcenters(self) -> int:
        """Return the number of subcenters used by the head, if exposed.

        Returns:
            Number of subcenters, or ``0`` when unavailable.
        """
        if hasattr(self.head, "num_subcenters"):
            return self.head.num_subcenters
        else:
            return 0

    @property
    def loss_type(self) -> Any:
        """Return the loss type reported by the head.

        Returns:
            Loss type value reported by the active head.
        """
        if hasattr(self.head, "loss_type"):
            return self.head.loss_type
        else:
            raise ValueError("head has no loss_type attribute")

    def has_param_groups(self) -> bool:
        """Return whether the model exposes custom optimizer parameter groups.

        Returns:
            ``True`` when custom optimizer parameter groups are defined.
        """
        return (
            super().has_param_groups()
            or self.pooling_weight_decay is not None
            or self.proj_weight_decay is not None
            or self.head_weight_decay is not None
        )

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups for the trainable components.

        Returns:
            Parameter groups with optional component-specific weight decay.
        """
        if (
            self.pooling_weight_decay is None
            and self.proj_weight_decay is None
            and self.head_weight_decay is None
        ):
            return super().trainable_param_groups()

        pooling = []
        proj_head = []
        head = []
        other = []
        bias = []
        for name, param in self.trainable_named_parameters():
            # we do not regularize biases nor Norm parameters
            if self.bias_weight_decay is not None and (
                name.endswith(".bias") or len(param.shape) == 1
            ):
                bias.append(param)
            else:
                if self.pooling_weight_decay is not None and name.startswith("pooling"):
                    pooling.append(param)
                elif self.proj_weight_decay is not None and name.startswith(
                    "proj_head"
                ):
                    proj_head.append(param)
                elif self.head_weight_decay is not None and name.startswith("head"):
                    head.append(param)
                else:
                    other.append(param)

        trainable_params = []
        if other:
            trainable_params.append({"params": other})
        if pooling:
            trainable_params.append(
                {"params": pooling, "weight_decay": self.pooling_weight_decay}
            )
        if proj_head:
            trainable_params.append(
                {"params": proj_head, "weight_decay": self.proj_weight_decay}
            )
        if head:
            trainable_params.append(
                {"params": head, "weight_decay": self.head_weight_decay}
            )
        if bias:
            trainable_params.append(
                {"params": bias, "weight_decay": self.bias_weight_decay}
            )

        return trainable_params

    def update_train_length(self, input_length: int) -> None:
        """Update the maximum input length seen during training.

        Args:
            input_length: Length of the current input sequence.
        """
        if not self.training:
            return
        if input_length > int(self.max_input_length.item()):
            # Keep the buffer registered by mutating the tensor rather than reassigning.
            self.max_input_length.fill_(input_length)

    def update_loss_margin(self, global_step: int) -> None:
        """Update margin scheduling for large-margin losses when supported.

        Args:
            global_step: Current optimisation step (or epoch) used to drive the
                scheduler.
        """
        if hasattr(self.head, "update_margin"):
            self.head.update_margin(global_step)

    def update_hyperparams(self, global_step: int) -> None:
        """Refresh any head hyperparameters that evolve during training.

        Args:
            global_step: Current optimisation step (or epoch).
        """
        self.update_loss_margin(global_step)

    def init_from_xvector(self, xvector_model: HyperTorchModel) -> None:
        """Initialize x-vector model backbone parameters from a pre-trained x-vector model.

        Args:
            xvector_model: Pre-trained x-vector model to use for initialization.
        """
        raise NotImplementedError()

    def forward_backbone(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_hidden_feats: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[Union[List[torch.Tensor], torch.Tensor]],
    ]:
        """Compute backbone features for the provided input signal.

        Args:
            x: Input waveform tensor with shape ``(batch, samples)``.
            x_lengths: Optional integer tensor with valid waveform sample counts.
            return_hidden_feats: When ``True``, subclasses should also return hidden
                feature maps from intermediate backbone layers.

        Returns:
            Tuple containing backbone output features shaped ``(batch, time, features)``,
            their lengths, the hidden
            feature maps (if requested), and the corresponding lengths.

        Raises:
            NotImplementedError: Subclasses must supply a concrete implementation.
        """
        raise NotImplementedError("forward_backbone is not implemented")

    def forward_adapter(
        self,
        backbone_output_feats: torch.Tensor,
        backbone_output_feats_lengths: Optional[torch.Tensor] = None,
        backbone_hidden_feats: Optional[List[torch.Tensor]] = None,
        backbone_hidden_feats_lengths: Optional[
            Union[List[torch.Tensor], torch.Tensor]
        ] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[Union[List[torch.Tensor], torch.Tensor]],
    ]:
        """Adapt backbone outputs before global pooling.

        Args:
            backbone_output_feats: Backbone features shaped ``(batch, time, features)``.
            backbone_output_feats_lengths: Optional sequence lengths associated with
                ``backbone_output_feats``.
            backbone_hidden_feats: Optional list of hidden feature maps captured inside
                the backbone.
            backbone_hidden_feats_lengths: Optional shared lengths tensor for all
                hidden features, or per-hidden-feature length tensors.

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
        compute_xvector_sig_reg: bool = True,
    ) -> XVectorPOutput:
        """Run a forward pass through the x-vector pipeline.

        Args:
            audio: Input tensor with shape ``(batch, samples)``.
            audio_lengths: Optional integer tensor with valid sample counts in ``audio``.
            target: Optional class labels used when the head computes a loss.
            target_mask: Optional boolean tensor indicating which targets are valid.
            return_backbone_feats: When ``True``, include backbone features in the
                returned payload.
            return_head_output: When ``True``, compute and return the Hydra head
                output (logits/loss or regression predictions).
            compute_xvector_sig_reg: Calculate the enabled embedding regularizer.
                Chunked inference defers it until embeddings have been aggregated.

        Returns:
            Structured output containing the embedding, optional head output, and
            requested backbone features.
        """
        self.update_train_length(audio.size(-1))
        with self._backbone_context:
            (
                backbone_output_feats,
                backbone_output_feats_lengths,
                backbone_hidden_feats,
                backbone_hidden_feats_lengths,
            ) = self.forward_backbone(
                audio,
                audio_lengths,
                return_hidden_feats=return_backbone_feats,
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

        with self._pooling_context:
            pooled_feats = self.pooling(
                backbone_output_feats.transpose(1, 2), backbone_output_feats_lengths
            )

        with self._proj_context:
            xvector = self.proj_head(pooled_feats)
        if return_head_output and self.head is not None:
            head_output = self.head(xvector, target, target_mask)
        else:
            head_output = None

        output = XVectorPOutput(
            xvector=xvector,
            xvector_sig_reg=(
                self.xvector_sig_reg(xvector)
                if self.enable_xvector_sig_reg and compute_xvector_sig_reg
                else None
            ),
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

    @staticmethod
    def _split_batches(
        tensor: torch.Tensor,
        lengths_tensor: Optional[torch.Tensor],
        chunk_length: int,
        max_batch_length: Optional[int],
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
        """Split tensors into batches that satisfy duration constraints.

        Args:
            tensor: Chunked waveform tensor.
            lengths_tensor: Optional valid sample counts per chunk.
            chunk_length: Number of samples in each padded chunk.
            max_batch_length: Optional maximum total samples per batch.

        Returns:
            Waveform batches and corresponding optional length batches.
        """
        if tensor.size(0) == 0:
            if lengths_tensor is None:
                return [], None
            return [], []

        if max_batch_length is None:
            audio_batches = [tensor]
            lengths_batches = [lengths_tensor] if lengths_tensor is not None else None
            return audio_batches, lengths_batches

        max_chunks_per_batch = max(1, max_batch_length // max(1, chunk_length))
        audio_batches: List[torch.Tensor] = []
        lengths_batches: Optional[List[torch.Tensor]]
        if lengths_tensor is None:
            lengths_batches = None
        else:
            lengths_batches = []

        for start in range(0, tensor.size(0), max_chunks_per_batch):
            end = min(start + max_chunks_per_batch, tensor.size(0))
            audio_batches.append(tensor[start:end])
            if lengths_batches is not None and lengths_tensor is not None:
                lengths_batches.append(lengths_tensor[start:end])

        return audio_batches, lengths_batches

    def _prepare_infer_input(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_batch_duration: Optional[float] = None,
        override_chunk_duration: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]], torch.Tensor]:
        """Prepare input audio for inference.

        Args:
            audio: Input tensor with shape ``(batch, time)``.
            audio_lengths: Optional sequence-length tensor describing valid samples in
                ``audio``.
            max_batch_duration: Optional maximum duration (in seconds) for batching.
            override_chunk_duration: Optional chunk duration (in seconds) to override
                any internal chunking mechanism.

        Returns:
            Tuple containing the prepared audio tensors grouped into batches (each
            respecting the maximum batch duration when provided), optional lists of
            adjusted lengths tensors aligned with each batch, and a mapping from every
            chunked element to its originating example.
        """
        if audio.ndim != 2 or audio.size(0) == 0 or audio.size(1) == 0:
            raise ValueError(
                "audio must have shape (batch, samples) with nonempty dimensions"
            )
        if audio_lengths is not None:
            if (
                audio_lengths.dtype == torch.bool
                or torch.is_floating_point(audio_lengths)
                or torch.is_complex(audio_lengths)
            ):
                raise TypeError("audio_lengths must contain integer sample counts")
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            if (
                audio_lengths.shape != (audio.size(0),)
                or torch.any(audio_lengths <= 0)
                or torch.any(audio_lengths > audio.size(1))
            ):
                raise ValueError(
                    "audio_lengths must contain a positive valid length per example"
                )

        if max_batch_duration is not None:
            if not math.isfinite(max_batch_duration) or max_batch_duration <= 0:
                raise ValueError("max_batch_duration must be finite and positive")
            max_batch_length = int(max_batch_duration * self.sample_frequency)
            if max_batch_length <= 0:
                raise ValueError(
                    "max_batch_duration must correspond to at least one sample"
                )
        else:
            max_batch_length = None

        if override_chunk_duration is not None:
            if (
                not math.isfinite(override_chunk_duration)
                or override_chunk_duration <= 0
            ):
                raise ValueError("override_chunk_duration must be finite and positive")
            chunk_length = int(override_chunk_duration * self.sample_frequency)
        else:
            if self.requires_max_train_length:
                if self.max_chunk_length <= 0:
                    raise ValueError(
                        "Model requires a positive max_chunk_length but it is not set. "
                        "Please provide override_chunk_duration or ensure max_chunk_length is set during training."
                    )
                chunk_length = self.max_chunk_length
            else:
                chunk_length = audio.size(-1)

        if max_batch_length is not None and max_batch_length < chunk_length:
            chunk_length = max_batch_length

        if chunk_length <= 0:
            raise ValueError("chunk_length must be a positive integer")

        batch_size = audio.size(0)
        time_dim = audio.size(-1)
        num_chunks = max(1, math.ceil(time_dim / chunk_length))
        chunk_length = max(1, math.ceil(time_dim / num_chunks))
        padded_length = num_chunks * chunk_length
        if padded_length > time_dim:
            audio = F.pad(audio, (0, padded_length - time_dim))

        audio = audio.reshape(batch_size * num_chunks, chunk_length)

        audio_index = torch.arange(
            batch_size, device=audio.device, dtype=torch.long
        ).repeat_interleave(num_chunks)

        if audio_lengths is None:
            audio_lengths = torch.full(
                (batch_size,), time_dim, device=audio.device, dtype=torch.long
            )

        lengths_list = [int(length) for length in audio_lengths.tolist()]
        keep_mask: List[bool] = []
        chunk_lengths: List[int] = []
        for length in lengths_list:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_length
                valid = min(max(length - start, 0), chunk_length)
                keep_mask.append(valid > 0)
                if valid > 0:
                    chunk_lengths.append(valid)

        keep_mask_tensor = torch.tensor(
            keep_mask, device=audio.device, dtype=torch.bool
        )
        assert (
            keep_mask_tensor.any()
        ), "Expected at least one chunk with positive length."
        audio = audio[keep_mask_tensor]
        audio_index = audio_index[keep_mask_tensor]
        new_audio_lengths = torch.tensor(
            chunk_lengths,
            device=audio_lengths.device,
            dtype=audio_lengths.dtype,
        )

        audio_batches, audio_lengths_batches = self._split_batches(
            audio, new_audio_lengths, chunk_length, max_batch_length
        )
        return audio_batches, audio_lengths_batches, audio_index

    def infer(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_batch_duration: Optional[float] = None,
        override_chunk_duration: Optional[float] = None,
        return_head_output: bool = False,
    ) -> XVectorPOutput:
        """Run inference through the x-vector pipeline.

        Args:
            audio: Input tensor with shape ``(batch, time)``.
            audio_lengths: Optional sequence-length tensor describing valid samples in
                ``audio``.
            max_batch_duration: Optional limit (in seconds) for the aggregated batch
                duration.
            override_chunk_duration: Optional override for the internal chunk length
                (in seconds).
            return_head_output: When ``True``, include the head output in the
                resulting :class:`XVectorPOutput` structure.

        Returns:
            XVectorPOutput: Weighted-average x-vector output aggregating all chunked
            batches for each original example.
        """
        audio_batches, audio_lengths_batches, audio_index = self._prepare_infer_input(
            audio, audio_lengths, max_batch_duration, override_chunk_duration
        )
        device = next(self.parameters()).device
        processed_lengths_batches: Optional[List[torch.Tensor]]
        if audio_lengths_batches is None:
            processed_lengths_batches = None
        else:
            processed_lengths_batches = []

        outputs: List[XVectorPOutput] = []
        for idx, audio_batch in enumerate(audio_batches):
            audio_batch = audio_batch.to(device)
            batch_lengths = (
                None
                if audio_lengths_batches is None
                else audio_lengths_batches[idx].to(device)
            )
            if processed_lengths_batches is not None and batch_lengths is not None:
                processed_lengths_batches.append(batch_lengths)
            output = self.forward(
                audio_batch,
                batch_lengths,
                return_backbone_feats=False,
                return_head_output=return_head_output,
                compute_xvector_sig_reg=False,
            )
            outputs.append(output)

        assert (
            len(outputs) > 0
        ), "_prepare_infer_input should always produce at least one batch"
        concatenated_output = XVectorPOutput.concatenate(outputs)

        if processed_lengths_batches is not None and processed_lengths_batches:
            chunk_weights = torch.cat(processed_lengths_batches, dim=0).to(
                concatenated_output.xvector.device,
                dtype=concatenated_output.xvector.dtype,
            )
        else:
            chunk_weights = None

        aggregated_output = XVectorPOutput.weighted_average_by_index(
            concatenated_output,
            audio_index.to(concatenated_output.xvector.device),
            chunk_weights,
        )
        if self.enable_xvector_sig_reg:
            # SIGReg is nonlinear in the sample distribution: recompute for the
            # returned embeddings rather than averaging per-chunk statistics.
            aggregated_output.xvector_sig_reg = self.xvector_sig_reg(
                aggregated_output.xvector
            )
        return aggregated_output

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the constructor arguments.

        Args:
            no_class_name: Whether to omit the registered model class name.

        Returns:
            Dict[str, Any]: Configuration dictionary that can be fed back into the
            constructor (along with subclass-specific backbone parameters).
        """
        head = (
            {
                **self.head.get_config(no_class_name=True),
                "head_type": self.head.head_type,
            }
            if self.head is not None
            else {"head_type": "none"}
        )
        config = {
            "pooling": PF.get_config(self.pooling),
            "xvector_dim": self.xvector_dim,
            "proj_use_norm": self.proj_use_norm,
            "proj_norm_layer": self.proj_norm_layer,
            "proj_norm_before": self.proj_norm_before,
            "enable_xvector_sig_reg": self.enable_xvector_sig_reg,
            "xvector_sig_reg": dict(self.xvector_sig_reg_args),
            "head": head,
            "pooling_weight_decay": self.pooling_weight_decay,
            "proj_weight_decay": self.proj_weight_decay,
            "head_weight_decay": self.head_weight_decay,
            "bias_weight_decay": self.bias_weight_decay,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        base_config.update(config)
        return base_config

    def change_config(
        self,
        xvector_dim: Optional[int] = None,
        override_head: bool = False,
        head: Optional[Union[Dict[str, Any], HydraHead]] = None,
        override_sig_reg: bool = False,
        enable_xvector_sig_reg: bool = False,
        xvector_sig_reg: Optional[Dict[str, Any]] = None,
        proj_use_norm: Optional[bool] = None,
        proj_norm_layer: Optional[str] = None,
        proj_norm_before: Optional[bool] = None,
        pooling_weight_decay: Optional[float] = None,
        proj_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Override embedding, projection, head, or optimizer settings for fine-tuning.

        Args:
            xvector_dim: New embedding dimension; rebuilds the projection and head.
            override_head: Whether to replace or reconfigure the downstream head.
            head: Head configuration or module required when overriding the head.
            override_sig_reg: Replace or disable the embedding regularizer when true.
            enable_xvector_sig_reg: Enable the replacement regularizer; otherwise disable it.
            xvector_sig_reg: Constructor settings for the replacement SIGReg module.
            proj_use_norm: Whether to enable projection normalization.
            proj_norm_layer: New normalization type; ``None`` keeps the current type.
            proj_norm_before: Whether normalization precedes projection.
            pooling_weight_decay: New pooling weight decay.
            proj_weight_decay: New projection weight decay.
            head_weight_decay: New downstream head weight decay.
            bias_weight_decay: New bias and normalization weight decay.
        """
        if override_head and head is None:
            raise ValueError("head must be provided when override_head=True")

        if override_sig_reg:
            sig_reg_args = dict(xvector_sig_reg or {})
            if enable_xvector_sig_reg:
                sig_reg_args["distributed_mode"] = "global_data"
                regularizer = SIGReg(**sig_reg_args).to(
                    device=self.proj_head.proj.weight.device
                )
                regularizer.train(self.training)
                self.xvector_sig_reg = regularizer
            elif hasattr(self, "xvector_sig_reg"):
                del self.xvector_sig_reg
            self.enable_xvector_sig_reg = enable_xvector_sig_reg
            self.xvector_sig_reg_args = dict(xvector_sig_reg or {})

        dim_changed = xvector_dim is not None and xvector_dim != self.xvector_dim
        proj_options = {
            "proj_use_norm": proj_use_norm,
            "proj_norm_layer": proj_norm_layer,
            "proj_norm_before": proj_norm_before,
        }
        proj_changed = any(
            value is not None and value != getattr(self, name)
            for name, value in proj_options.items()
        )
        if dim_changed or proj_changed:
            old_proj = self.proj_head
            for name, value in proj_options.items():
                if value is not None:
                    setattr(self, name, value)
            if xvector_dim is not None:
                self.xvector_dim = xvector_dim
            self.proj_head = ProjHead(
                in_feats=old_proj.in_feats,
                out_feats=self.xvector_dim,
                norm_layer=self.proj_norm_layer,
                use_norm=self.proj_use_norm,
                norm_before=self.proj_norm_before,
            ).to(device=old_proj.proj.weight.device, dtype=old_proj.proj.weight.dtype)
            if not dim_changed:
                with torch.no_grad():
                    self.proj_head.proj.weight.copy_(old_proj.proj.weight)
                    if (
                        self.proj_head.proj.bias is not None
                        and old_proj.proj.bias is not None
                    ):
                        self.proj_head.proj.bias.copy_(old_proj.proj.bias)
                if (
                    self.proj_head.use_norm
                    and old_proj.use_norm
                    and self.proj_head.norm_layer == old_proj.norm_layer
                    and self.proj_head.norm_before == old_proj.norm_before
                ):
                    self.proj_head._norm_layer.load_state_dict(
                        old_proj._norm_layer.state_dict()
                    )
            if dim_changed and not override_head and self.head is not None:
                head_config = self.head.get_config(no_class_name=True)
                head_config["in_feats"] = self.xvector_dim
                self.head = type(self.head)(**head_config).to(
                    device=old_proj.proj.weight.device, dtype=old_proj.proj.weight.dtype
                )

        if override_head:
            if isinstance(head, HydraHead):
                self.head = head
            else:
                head_config = {**head, "in_feats": self.xvector_dim}
                self.head = (
                    HydraHeadFactory.create(**head_config)
                    if self.head is None or head_config.get("head_type") == "none"
                    else HydraHeadFactory.reconfig_or_create(self.head, **head_config)
                )
            if self.head is not None:
                self.head.to(
                    device=self.proj_head.proj.weight.device,
                    dtype=self.proj_head.proj.weight.dtype,
                )

        if bias_weight_decay is not None:
            logging.info(
                f"overriding bias weight decay with new value: {bias_weight_decay}"
            )
            self.bias_weight_decay = bias_weight_decay

        if pooling_weight_decay is not None:
            logging.info(
                f"overriding pooling weight decay with new value: {pooling_weight_decay}"
            )
            self.pooling_weight_decay = pooling_weight_decay

        if proj_weight_decay is not None:
            logging.info(
                f"overriding proj head weight decay with new value: {proj_weight_decay}"
            )
            self.proj_weight_decay = proj_weight_decay

        if head_weight_decay is not None:
            logging.info(
                f"overriding head weight decay with new value: {head_weight_decay}"
            )
            self.head_weight_decay = head_weight_decay

        if dim_changed or proj_changed or override_head:
            # Rebuilt components must inherit the active freeze and eval settings.
            train_mode = self._train_mode
            training = self.training
            self._train_mode = None
            self.set_train_mode(train_mode)
            self.train(training)

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Return supported training regimes.

        Returns:
            Training-mode strings for x-vector+ models.
        """
        return XVectorPTrainMode.choices()

    def set_train_mode(self, mode: Union[str, XVectorPTrainMode]) -> None:
        """Set which model components can receive gradients.

        Args:
            mode: Full, frozen, frozen feature extractor, pooling, projection,
                or output head training mode.
        """
        mode = XVectorPTrainMode(mode).value
        if mode == self._train_mode:
            return
        self._backbone_context = contextlib.nullcontext()
        self._adapter_context = contextlib.nullcontext()
        self._pooling_context = contextlib.nullcontext()
        self._proj_context = contextlib.nullcontext()
        if mode == XVectorPTrainMode.FULL:
            self.unfreeze()
        elif mode == XVectorPTrainMode.FROZEN:
            self.freeze()
        elif mode == XVectorPTrainMode.FROZEN_FEAT_EXTRACTOR:
            self.unfreeze()
            self.freeze_backbone_feat_extractor()
        else:
            self.freeze()
            self._backbone_context = torch.no_grad()
            self._adapter_context = torch.no_grad()
            if mode == XVectorPTrainMode.POOLING:
                for param in self.pooling.parameters():
                    param.requires_grad_(True)
            else:
                self._pooling_context = torch.no_grad()
            if mode != XVectorPTrainMode.OUTPUT_LAYER:
                self.proj_head.unfreeze()
            else:
                self._proj_context = torch.no_grad()
            if self.head is not None:
                self.head.unfreeze()
        self._train_mode = mode
        if self.training:
            self.train()

    def _train(self, train_mode: Union[str, XVectorPTrainMode]) -> None:
        """Apply component training/evaluation states for the active regime.

        Args:
            train_mode: Target x-vector+ training regime.
        """
        train_mode = XVectorPTrainMode(train_mode).value
        super()._train("frozen" if train_mode == "frozen" else "full")
        if train_mode in ("full", "frozen"):
            return
        if train_mode == XVectorPTrainMode.FROZEN_FEAT_EXTRACTOR:
            self.set_backbone_feat_extractor_in_eval_mode()
            return
        self.set_backbone_in_eval_mode()
        self.set_adapters_in_eval_mode()
        self.pooling.train(train_mode == XVectorPTrainMode.POOLING)
        self.proj_head.train(train_mode != XVectorPTrainMode.OUTPUT_LAYER)

    def freeze_backbone_feat_extractor(self) -> None:
        """Freeze backbone feature extractor parameters."""
        raise NotImplementedError("freeze_backbone_feat_extractor is not implemented")

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        raise NotImplementedError("freeze_backbone is not implemented")

    def freeze_adapters(self) -> None:
        """Freeze adapter modules."""
        raise NotImplementedError("freeze_adapters is not implemented")

    def set_backbone_feat_extractor_in_train_mode(self) -> None:
        """Put the backbone feature extractor into training mode."""
        raise NotImplementedError(
            "set_backbone_feat_extractor_in_train_mode not implemented"
        )

    def set_backbone_feat_extractor_in_eval_mode(self) -> None:
        """Put the backbone feature extractor into evaluation mode."""
        raise NotImplementedError(
            "set_backbone_feat_extractor_in_eval_mode not implemented"
        )

    def set_backbone_in_train_mode(self) -> None:
        """Put the backbone into training mode."""
        raise NotImplementedError("set_backbone_in_train_mode not implemented")

    def set_backbone_in_eval_mode(self) -> None:
        """Put the backbone into evaluation mode."""
        raise NotImplementedError("set_backbone_in_eval_mode not implemented")

    def set_adapters_in_train_mode(self) -> None:
        """Put adapter modules into training mode."""
        raise NotImplementedError("set_adapters_in_train_mode not implemented")

    def set_adapters_in_eval_mode(self) -> None:
        """Put adapter modules into evaluation mode."""
        raise NotImplementedError("set_adapters_in_eval_mode not implemented")

    def compute_prototype_affinity(self) -> torch.Tensor:
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
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Return constructor-compatible keyword arguments.

        Args:
            **kwargs: Candidate model configuration values.

        Returns:
            Keyword arguments accepted by ``XVectorP.__init__``.
        """
        return filter_func_args(XVectorP.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI/configuration arguments for XVectorP models.

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

        if "xvector_dim" not in skip:
            parser.add_argument(
                "--xvector-dim",
                type=int,
                default=256,
                help="final x-vector embedding dimension",
            )
        if "proj_use_norm" not in skip:
            parser.add_argument(
                "--proj-use-norm",
                default=True,
                action=ActionYesNo,
                help="enable normalization of the projection input or output",
            )
        if "proj_norm_layer" not in skip:
            parser.add_argument(
                "--proj-norm-layer",
                default=None,
                type=str,
                choices=["batch-norm", "layer-norm", "rms-norm"],
                help="projection normalization type (batch-norm by default)",
            )
        if "proj_norm_before" not in skip:
            parser.add_argument(
                "--proj-norm-before",
                default=True,
                action=ActionYesNo,
                help="apply normalization before projection; false applies it after",
            )
        if "bias_weight_decay" not in skip:
            parser.add_argument(
                "--bias-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for biases and normalization parameters",
            )
        if "pooling_weight_decay" not in skip:
            parser.add_argument(
                "--pooling-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for global pooling parameters",
            )
        if "enable_xvector_sig_reg" not in skip:
            parser.add_argument(
                "--enable-xvector-sig-reg",
                default=False,
                action=ActionYesNo,
                help="Calculate SIGReg on projected xvectors.",
            )
        if "xvector_sig_reg" not in skip:
            SIGReg.add_class_args(parser, prefix="xvector_sig_reg", skip=skip)

        if "proj_weight_decay" not in skip:
            parser.add_argument(
                "--proj-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for projection-head parameters",
            )
        if "head_weight_decay" not in skip:
            parser.add_argument(
                "--head-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for downstream head parameters",
            )

        if "pooling" not in skip:
            PF.add_class_args(parser, prefix="pooling")

        if "head" not in skip:
            HydraHeadFactory.add_class_args(
                parser,
                prefix="head",
                skip=skip,
            )

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Return fine-tuning keyword arguments accepted by ``change_config``.

        Args:
            **kwargs: Candidate fine-tuning configuration values.

        Returns:
            Keyword arguments accepted by ``XVectorP.change_config``.
        """
        args = filter_func_args(XVectorP.change_config, kwargs)
        return args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register fine-tuning overrides for XVectorP models.

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

        if "override_sig_reg" not in skip:
            parser.add_argument(
                "--override-sig-reg",
                default=False,
                action=ActionYesNo,
                help="replace or disable the saved x-vector SIGReg configuration",
            )
        if "enable_xvector_sig_reg" not in skip:
            parser.add_argument(
                "--enable-xvector-sig-reg",
                default=False,
                action=ActionYesNo,
                help="enable x-vector SIGReg when overriding its configuration",
            )
        if "xvector_sig_reg" not in skip:
            SIGReg.add_class_args(parser, prefix="xvector_sig_reg", skip=skip)

        if "xvector_dim" not in skip:
            parser.add_argument(
                "--xvector-dim",
                type=int,
                default=None,
                help="override embedding dimension, rebuilding the projection and downstream head",
            )

        if "proj_use_norm" not in skip:
            parser.add_argument(
                "--proj-use-norm",
                default=None,
                action=ActionYesNo,
                help="enable normalization of the projection input or output",
            )
        if "proj_norm_layer" not in skip:
            parser.add_argument(
                "--proj-norm-layer",
                default=None,
                type=str,
                choices=["batch-norm", "layer-norm", "rms-norm"],
                help="override projection normalization type (unchanged by default)",
            )
        if "proj_norm_before" not in skip:
            parser.add_argument(
                "--proj-norm-before",
                default=None,
                action=ActionYesNo,
                help="apply normalization before projection; false applies it after",
            )
        if "bias_weight_decay" not in skip:
            parser.add_argument(
                "--bias-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for biases and normalization parameters",
            )
        if "pooling_weight_decay" not in skip:
            parser.add_argument(
                "--pooling-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for global pooling parameters",
            )
        if "proj_weight_decay" not in skip:
            parser.add_argument(
                "--proj-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for projection-head parameters",
            )
        if "head_weight_decay" not in skip:
            parser.add_argument(
                "--head-weight-decay",
                type=float,
                default=None,
                help="optional weight decay override for downstream head parameters",
            )

        if "override_head" not in skip:
            parser.add_argument(
                "--override-head",
                default=False,
                action=ActionYesNo,
                help="replace or reconfigure the downstream head using the head options",
            )

        if "head" not in skip:
            HydraHeadFactory.add_class_args(
                parser,
                prefix="head",
                skip=skip,
            )

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

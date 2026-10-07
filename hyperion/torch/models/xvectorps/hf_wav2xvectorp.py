"""
Copyright 2026 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs import FeatFuserMVN, HydraHead
from .xvectorp import XVectorP


class HFWav2XVectorP(XVectorP):
    """Pooled x-vector+ model using a Hugging Face speech backbone.

    The final hidden state is used directly by default. An optional feature fuser
    can combine a selected range of hidden states before global pooling.

    Attributes:
        hf_feats: Hugging Face waveform feature extractor and encoder.
        feat_fuser: Optional module that fuses selected hidden states.
        feat_fusion_start: First hidden-state index passed to the feature fuser.
        backbone_lr: Optional learning-rate override for encoder parameters.
        backbone_weight_decay: Optional weight-decay override for encoder parameters.
        backbone_feats_lr: Optional learning-rate override for feature-extractor parameters.
        backbone_feats_weight_decay: Optional weight-decay override for feature-extractor parameters.
        pooling: Global pooling module (inherited).
        proj_head: Projection head (inherited).
        head: Optional downstream classification or regression head (inherited).
    """

    def __init__(
        self,
        hf_feats: nn.Module,
        pooling: Union[str, Dict[str, Any], nn.Module],
        xvector_dim: int,
        head: Optional[Union[Dict[str, Any], HydraHead]],
        feat_fuser: Optional[Union[Dict[str, Any], FeatFuserMVN]] = None,
        feat_fusion_start: int = 0,
        proj_norm_layer: Optional[str] = None,
        proj_use_norm: bool = True,
        proj_norm_before: bool = True,
        enable_xvector_sig_reg: bool = False,
        xvector_sig_reg: Optional[Dict[str, Any]] = None,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        pooling_weight_decay: Optional[float] = None,
        proj_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialize the Hugging Face x-vector+ model.

        Args:
            hf_feats: Hugging Face waveform feature/backbone module.
            pooling: Global-pooling type, configuration, or module.
            xvector_dim: Dimension of the final embedding.
            head: Hydra head configuration or module, or ``None`` for embedding-only use.
            feat_fuser: Optional hidden-state fuser configuration or module.
            feat_fusion_start: Index of the first hidden state used by the fuser.
            proj_norm_layer: Projection normalization type; ``None`` selects batch norm.
            proj_use_norm: Whether projection normalization is enabled.
            proj_norm_before: Whether normalization precedes the projection.
            backbone_feats_lr: Optional feature-extractor learning-rate override.
            backbone_feats_weight_decay: Optional feature-extractor weight decay.
            backbone_lr: Optional encoder learning-rate override.
            backbone_weight_decay: Optional encoder weight decay.
            enable_xvector_sig_reg: Whether to calculate SIGReg on projected embeddings.
            xvector_sig_reg: SIGReg constructor arguments.
            pooling_weight_decay: Optional weight decay for global pooling.
            proj_weight_decay: Optional weight decay for the projection head.
            head_weight_decay: Optional weight decay for the downstream head.
            bias_weight_decay: Optional weight decay for bias and normalization parameters.
        """
        if isinstance(hf_feats, dict):
            raise TypeError("hf_feats must be a constructed Hugging Face module")

        self.feat_fusion_start = feat_fusion_start
        hidden_size = hf_feats.hidden_size
        fuser_config = (
            feat_fuser.feat_fuser_cfg
            if isinstance(feat_fuser, FeatFuserMVN)
            else feat_fuser
        )
        if fuser_config is not None:
            hidden_size = (
                fuser_config.get("feat_fuser", {}).get("proj_dim") or hidden_size
            )
        self._backbone_output_size = hidden_size
        super().__init__(
            pooling=pooling,
            xvector_dim=xvector_dim,
            head=head,
            proj_norm_layer=proj_norm_layer,
            proj_use_norm=proj_use_norm,
            proj_norm_before=proj_norm_before,
            enable_xvector_sig_reg=enable_xvector_sig_reg,
            xvector_sig_reg=xvector_sig_reg,
            pooling_weight_decay=pooling_weight_decay,
            proj_weight_decay=proj_weight_decay,
            head_weight_decay=head_weight_decay,
            bias_weight_decay=bias_weight_decay,
        )
        self.hf_feats = hf_feats
        self.feat_fuser = self._make_fuser(feat_fuser)
        self._hf_context = contextlib.nullcontext()
        self.backbone_feats_lr = backbone_feats_lr
        self.backbone_feats_weight_decay = backbone_feats_weight_decay
        self.backbone_lr = backbone_lr
        self.backbone_weight_decay = backbone_weight_decay

    def _make_fuser(
        self, feat_fuser: Optional[Union[Dict[str, Any], FeatFuserMVN]]
    ) -> Optional[FeatFuserMVN]:
        """Build an optional hidden-state fuser with backbone dimensions.

        Args:
            feat_fuser: Fuser configuration, constructed module, or ``None``.

        Returns:
            Configured fuser module, or ``None`` when fusion is disabled.
        """
        if feat_fuser is None:
            return None
        if isinstance(feat_fuser, FeatFuserMVN):
            return feat_fuser

        config = deepcopy(feat_fuser)
        num_feats = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        if self.feat_fusion_start < 0 or num_feats <= 0:
            raise ValueError(
                "feat_fusion_start must select at least one hidden state; "
                f"got {self.feat_fusion_start} for {self.hf_feats.num_encoder_layers} encoder layers"
            )
        inner = config.setdefault("feat_fuser", {})
        inner["num_feats"] = num_feats
        inner["feat_dim"] = self.hf_feats.hidden_size
        return FeatFuserMVN(**config)

    def backbone_output_feats(self) -> int:
        """Return the number of features emitted per backbone frame.

        Returns:
            Feature dimension after optional hidden-state fusion.
        """
        return self._backbone_output_size

    @property
    def sample_frequency(self) -> int:
        """Return the waveform sample frequency expected by the backbone."""
        return self.hf_feats.sample_frequency

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
        """Extract final or fused hidden-state features.

        Args:
            x: Waveforms shaped ``(batch, samples)``.
            x_lengths: Optional valid waveform lengths in samples.
            return_hidden_feats: Whether to return the selected backbone states.

        Returns:
            Backbone features shaped ``(batch, time, features)``, their lengths,
            optional hidden-state features, and matching hidden-state lengths.
        """
        return_hid_states = self.feat_fuser is not None or return_hidden_feats
        with self._hf_context:
            output = self.hf_feats(x, x_lengths, return_hid_states=return_hid_states)

        lengths = output["hidden_states_lengths"]
        if return_hid_states:
            hidden = list(output["hidden_states"])
            fused_input = hidden[self.feat_fusion_start :]
            if self.feat_fuser is not None:
                feats, lengths = self.feat_fuser(fused_input, lengths)
                if self.feat_fuser.trans:
                    feats = feats.transpose(1, 2).contiguous()
            else:
                feats = hidden[-1]
            hidden_output = hidden if return_hidden_feats else None
        else:
            feats = output["last_hidden_state"]
            hidden_output = None

        return feats, lengths, hidden_output, lengths if return_hidden_feats else None

    def freeze_backbone(self) -> None:
        """Freeze all Hugging Face backbone parameters."""
        self.hf_feats.freeze()

    def freeze_backbone_feat_extractor(self) -> None:
        """Freeze the Hugging Face waveform feature encoder."""
        self.hf_feats.freeze_feature_encoder()

    def set_backbone_feat_extractor_in_train_mode(self) -> None:
        """Put the Hugging Face waveform feature encoder in training mode."""
        self.hf_feats._hf_backbone_model().feature_extractor.train()

    def set_backbone_feat_extractor_in_eval_mode(self) -> None:
        """Keep the frozen waveform feature encoder in evaluation mode."""
        self.hf_feats._hf_backbone_model().feature_extractor.eval()

    def freeze_adapters(self) -> None:
        """Freeze the optional hidden-state fuser."""
        if self.feat_fuser is not None:
            self.feat_fuser.requires_grad_(False)

    def set_backbone_in_train_mode(self) -> None:
        """Put the Hugging Face backbone in training mode."""
        self.hf_feats.train()

    def set_backbone_in_eval_mode(self) -> None:
        """Put the Hugging Face backbone in evaluation mode."""
        self.hf_feats.eval()

    def set_adapters_in_train_mode(self) -> None:
        """Put the optional hidden-state fuser in training mode."""
        if self.feat_fuser is not None:
            self.feat_fuser.train()

    def set_adapters_in_eval_mode(self) -> None:
        """Put the optional hidden-state fuser in evaluation mode."""
        if self.feat_fuser is not None:
            self.feat_fuser.eval()

    def has_param_groups(self) -> bool:
        """Return whether custom Hugging Face or model parameter groups are needed.

        Returns:
            ``True`` if this model or the backbone has custom optimizer groups.
        """
        return (
            super().has_param_groups()
            or self.hf_feats.has_param_groups()
            or self.backbone_feats_lr is not None
            or self.backbone_feats_weight_decay is not None
            or self.backbone_lr is not None
            or self.backbone_weight_decay is not None
        )

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        """Return optimizer groups for the backbone and x-vector+ components.

        Returns:
            Parameter groups with configured component-specific overrides.
        """
        if not self.has_param_groups():
            return [{"params": self.trainable_parameters()}]

        backbone_bias_weight_decay = (
            self.bias_weight_decay
            if self.bias_weight_decay is not None
            else self.hf_feats.bias_weight_decay
        )
        separate_backbone_bias = backbone_bias_weight_decay is not None
        feat_params = list(
            self.hf_feats.trainable_feat_extract_params(bias=not separate_backbone_bias)
        )
        encoder_params = list(
            self.hf_feats.trainable_encoder_params(bias=not separate_backbone_bias)
        )
        feat_bias = (
            list(self.hf_feats.trainable_feat_extract_bias())
            if separate_backbone_bias
            else []
        )
        encoder_bias = (
            list(self.hf_feats.trainable_encoder_bias())
            if separate_backbone_bias
            else []
        )
        groups: List[Dict[str, Any]] = []
        if feat_params:
            group = {"params": feat_params}
            if self.backbone_feats_lr is not None:
                group["lr"] = self.backbone_feats_lr
            elif getattr(self.hf_feats, "feat_extract_lr", None) is not None:
                group["lr"] = self.hf_feats.feat_extract_lr
            if self.backbone_feats_weight_decay is not None:
                group["weight_decay"] = self.backbone_feats_weight_decay
            groups.append(group)
        if encoder_params:
            group = {"params": encoder_params}
            if self.backbone_lr is not None:
                group["lr"] = self.backbone_lr
            elif getattr(self.hf_feats, "encoder_lr", None) is not None:
                group["lr"] = self.hf_feats.encoder_lr
            if self.backbone_weight_decay is not None:
                group["weight_decay"] = self.backbone_weight_decay
            groups.append(group)

        if feat_bias:
            group = {"params": feat_bias, "weight_decay": backbone_bias_weight_decay}
            if self.backbone_feats_lr is not None:
                group["lr"] = self.backbone_feats_lr
            elif getattr(self.hf_feats, "feat_extract_lr", None) is not None:
                group["lr"] = self.hf_feats.feat_extract_lr
            groups.append(group)
        if encoder_bias:
            group = {
                "params": encoder_bias,
                "weight_decay": backbone_bias_weight_decay,
            }
            if self.backbone_lr is not None:
                group["lr"] = self.backbone_lr
            elif getattr(self.hf_feats, "encoder_lr", None) is not None:
                group["lr"] = self.hf_feats.encoder_lr
            groups.append(group)

        backbone_param_ids = {
            id(param) for group in groups for param in group["params"]
        }
        remaining = [
            (name, param)
            for name, param in self.trainable_named_parameters()
            if id(param) not in backbone_param_ids
        ]
        bias_params = []
        component_params: Dict[str, List[nn.Parameter]] = {
            "pooling": [],
            "proj_head": [],
            "head": [],
            "other": [],
        }
        for name, param in remaining:
            if self.bias_weight_decay is not None and (
                name.endswith(".bias") or param.ndim == 1
            ):
                bias_params.append(param)
            elif name.startswith("pooling") and self.pooling_weight_decay is not None:
                component_params["pooling"].append(param)
            elif name.startswith("proj_head") and self.proj_weight_decay is not None:
                component_params["proj_head"].append(param)
            elif name.startswith("head") and self.head_weight_decay is not None:
                component_params["head"].append(param)
            else:
                component_params["other"].append(param)

        for name, weight_decay in (
            ("pooling", self.pooling_weight_decay),
            ("proj_head", self.proj_weight_decay),
            ("head", self.head_weight_decay),
        ):
            if component_params[name]:
                groups.append(
                    {"params": component_params[name], "weight_decay": weight_decay}
                )
        if component_params["other"]:
            groups.append({"params": component_params["other"]})
        if bias_params:
            groups.append(
                {"params": bias_params, "weight_decay": self.bias_weight_decay}
            )
        return groups

    def change_config(
        self,
        hf_feats: Optional[Dict[str, Any]] = None,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update backbone optimizer settings and forward model overrides.

        Args:
            hf_feats: Optional Hugging Face backbone configuration changes.
            backbone_feats_lr: Optional feature-encoder learning-rate override.
            backbone_feats_weight_decay: Optional feature-encoder weight-decay override.
            backbone_lr: Optional Transformer encoder learning-rate override.
            backbone_weight_decay: Optional Transformer encoder weight-decay override.
            **kwargs: Projection, pooling, head, and other ``XVectorP`` overrides.
        """
        if hf_feats is not None:
            self.hf_feats.change_config(**hf_feats)
        for name, value in (
            ("backbone_feats_lr", backbone_feats_lr),
            ("backbone_feats_weight_decay", backbone_feats_weight_decay),
            ("backbone_lr", backbone_lr),
            ("backbone_weight_decay", backbone_weight_decay),
        ):
            if value is not None:
                logging.info("overriding %s with new value: %s", name, value)
                setattr(self, name, value)
        super().change_config(**kwargs)

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return constructor configuration for this Hugging Face model.

        Args:
            no_class_name: Whether to omit the registered model class name.

        Returns:
            JSON-serializable constructor settings.
        """
        config = super().get_config(no_class_name=no_class_name)
        config.update(
            hf_feats=self.hf_feats.get_config(no_class_name=True),
            feat_fuser=(
                None
                if self.feat_fuser is None
                else self.feat_fuser.get_config(no_class_name=True)
            ),
            feat_fusion_start=self.feat_fusion_start,
            backbone_feats_lr=self.backbone_feats_lr,
            backbone_feats_weight_decay=self.backbone_feats_weight_decay,
            backbone_lr=self.backbone_lr,
            backbone_weight_decay=self.backbone_weight_decay,
        )
        return config

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter a configuration to this wrapper's constructor arguments.

        Args:
            **kwargs: Candidate model configuration values.

        Returns:
            Constructor-compatible keyword arguments.
        """
        return filter_func_args(HFWav2XVectorP.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register Hugging Face x-vector+ configuration arguments.

        Args:
            parser: Parser receiving the model arguments.
            prefix: Optional namespace prefix.
            skip: Constructor argument names to omit.
        """
        skip = set(skip or ())
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        if "feat_fuser" not in skip:
            FeatFuserMVN.add_class_args(parser, prefix="feat_fuser")
        for name, default, help_text in (
            (
                "feat_fusion_start",
                0,
                "first hidden-state index used for feature fusion (0 selects the input embedding)",
            ),
            (
                "backbone_feats_lr",
                None,
                "learning-rate override for the HF feature encoder",
            ),
            (
                "backbone_feats_weight_decay",
                None,
                "weight-decay override for the HF feature encoder",
            ),
            (
                "backbone_lr",
                None,
                "learning-rate override for the HF Transformer encoder",
            ),
            (
                "backbone_weight_decay",
                None,
                "weight-decay override for the HF Transformer encoder",
            ),
        ):
            if name not in skip:
                parser.add_argument(
                    "--" + name.replace("_", "-"),
                    type=int if name == "feat_fusion_start" else float,
                    default=default,
                    help=help_text,
                )
        XVectorP.add_class_args(parser, skip=skip)
        if outer_parser is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter fine-tuning arguments for the HF wrapper and base model.

        Args:
            **kwargs: Candidate fine-tuning configuration values.

        Returns:
            Fine-tuning values accepted by the model and its wrapped backbone.
        """
        args = XVectorP.filter_finetune_args(**kwargs)
        args.update(filter_func_args(HFWav2XVectorP.change_config, kwargs))
        return args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register fine-tuning arguments for the HF wrapper.

        Args:
            parser: Parser receiving the fine-tuning options.
            prefix: Optional namespace prefix.
            skip: Constructor argument names to omit.
        """
        skip = set(skip or ())
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None
        for name in (
            "backbone_feats_lr",
            "backbone_feats_weight_decay",
            "backbone_lr",
            "backbone_weight_decay",
        ):
            if name not in skip:
                parser.add_argument(
                    "--" + name.replace("_", "-"),
                    type=float,
                    default=None,
                    help=f"override {name.replace('_', ' ')} during fine-tuning",
                )
        XVectorP.add_finetune_args(parser, skip=skip)
        if outer_parser is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

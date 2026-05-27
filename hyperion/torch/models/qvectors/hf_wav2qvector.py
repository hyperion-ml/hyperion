"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs import FeatFuserMVN, HydraHead, QFormerV2
from ...utils.masking import scale_seq_lengths
from .qvector import QVector


class HFWav2QVector(QVector):
    """Base q-vector model for HuggingFace-style encoder backbones.

    Attributes:
        hf_feats: Front-end/backbone module exposing HF-like hidden states.
        feat_fuser: Optional hidden-state fusion/mvn module configuration.
        hidden_feats_fusion_start: Index of the first hidden state used by the
            feature fuser when output features are aggregated.
        hidden_feats_agg_start: Index of the first hidden state used for hidden
            feature aggregation.
        hidden_feats_shared_adapters: Whether hidden-feature adapters are shared
            across selected hidden layers.
        backbone_layers: Indices of intermediate backbone layers returned during
            hidden-feature aggregation, or ``None`` when not requested.
        backbone_return_output: Flag indicating whether ``forward_hid_feats`` also
            returns the final backbone output tensor.
        hidden_feats_adapter: Optional projection layers that align backbone hidden
            features with the hidden Q-former input dimensionality.
        output_feats_adapter: Optional projection mapping backbone outputs to the
            output Q-former input space.
        backbone_feats_lr: Optional learning-rate override for feature-extractor
            parameters from ``hf_feats``.
        backbone_feats_weight_decay: Optional weight-decay override for feature-
            extractor parameters from ``hf_feats``.
        backbone_lr: Optional learning-rate override for encoder/backbone
            parameters from ``hf_feats``.
        backbone_weight_decay: Optional weight-decay override for encoder/backbone
            parameters from ``hf_feats``.
        adapter_lr: Optional learning-rate override for adapter parameters.
        adapter_weight_decay: Optional weight-decay override for adapter
            parameters.
        num_hidden_feats_queries: Number of learned queries used for hidden feature
            aggregation (inherited from ``QVector``).
        num_output_feats_queries: Number of learned queries used for output feature
            aggregation (inherited from ``QVector``).
        qvector_dim: Dimensionality of the flattened q-vector embedding (inherited).
        hidden_feats_agg_qformer: Q-former module that attends to intermediate
            backbone activations (inherited).
        output_feats_agg_qformer: Q-former operating on backbone outputs (inherited).
        hidden_feats_queries: Learnable queries feeding the hidden Q-former
            (inherited).
        output_feats_queries: Learnable queries feeding the output Q-former
            (inherited).
        proj_head: Projection head that flattens the concatenated Q-former outputs
            (inherited).
        head: Downstream Hydra head that produces logits or regression estimates
            (inherited).
    """

    def __init__(
        self,
        hf_feats: nn.Module,
        feat_fuser: Optional[Dict[str, Any]],
        hidden_feats_agg_qformer: Union[Dict[str, Any], QFormerV2, None],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Union[Dict[str, Any], None],
        num_output_feats_queries: int,
        qvector_dim: int,
        head: Union[Dict[str, Any], HydraHead],
        proj_bias: bool = True,
        hidden_feats_fusion_start: int = 0,
        hidden_feats_agg_start: int = 0,
        hidden_feats_shared_adapters: bool = True,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        adapter_lr: Optional[float] = None,
        adapter_weight_decay: Optional[float] = None,
        qformer_weight_decay: Optional[float] = None,
        proj_head_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialise a HuggingFace-backbone q-vector model.

        Args:
            hf_feats: HF feature/backbone module.
            feat_fuser: Optional configuration used to build
                :class:`FeatFuserMVN`. Required when
                ``output_feats_agg_qformer`` is provided.
            hidden_feats_agg_qformer: Hidden Q-former configuration or module.
            num_hidden_feats_queries: Number of hidden queries used for aggregation.
            output_feats_agg_qformer: Output Q-former configuration or module.
            num_output_feats_queries: Number of output queries.
            qvector_dim: Size of the final q-vector embedding.
            head: Hydra head configuration or module.
            proj_bias: Whether the projection head linear layer includes a bias term.
            hidden_feats_fusion_start: First hidden-state index used for output
                feature fusion.
            hidden_feats_agg_start: First hidden-state index used for hidden
                feature aggregation.
            hidden_feats_shared_adapters: If ``True``, hidden feature adapters
                share weights across selected layers.
            backbone_feats_lr: Optional learning-rate override for feature/backbone
                front-end parameters.
            backbone_feats_weight_decay: Optional weight-decay override for
                feature/backbone front-end parameters.
            backbone_lr: Optional learning-rate override for fused-backbone
                parameters.
            backbone_weight_decay: Optional weight-decay override for
                fused-backbone parameters.
            adapter_lr: Optional learning-rate override for adapter parameters
                (``hidden_feats_adapter`` and ``output_feats_adapter``).
            adapter_weight_decay: Optional weight-decay override for adapter
                parameters (``hidden_feats_adapter`` and ``output_feats_adapter``).
            qformer_weight_decay: Optional weight-decay override applied to both
                hidden/output Q-former parameters.
            proj_head_weight_decay: Optional weight-decay override applied to
                projection-head parameters.
            head_weight_decay: Optional weight-decay override applied to downstream
                head parameters.
            bias_weight_decay: Optional weight decay applied only to bias parameters.
        """

        super().__init__(
            hidden_feats_agg_qformer=hidden_feats_agg_qformer,
            num_hidden_feats_queries=num_hidden_feats_queries,
            output_feats_agg_qformer=output_feats_agg_qformer,
            num_output_feats_queries=num_output_feats_queries,
            qvector_dim=qvector_dim,
            head=head,
            proj_bias=proj_bias,
            qformer_weight_decay=qformer_weight_decay,
            proj_head_weight_decay=proj_head_weight_decay,
            head_weight_decay=head_weight_decay,
            bias_weight_decay=bias_weight_decay,
        )
        self.hf_feats = hf_feats
        self.hidden_feats_fusion_start = hidden_feats_fusion_start
        self.hidden_feats_agg_start = hidden_feats_agg_start
        self.hidden_feats_shared_adapters = hidden_feats_shared_adapters
        self.backbone_lr = backbone_lr
        self.backbone_weight_decay = backbone_weight_decay
        self.backbone_feats_lr = backbone_feats_lr
        self.backbone_feats_weight_decay = backbone_feats_weight_decay
        self.adapter_lr = adapter_lr
        self.adapter_weight_decay = adapter_weight_decay
        self._hf_context = contextlib.nullcontext()
        self.backbone_layers: Optional[List[int]] = None
        if self.output_feats_agg_qformer is None:
            self.backbone_return_output: bool = False
        else:
            self.backbone_return_output: bool = True
        self.hidden_feats_adapter: Optional[nn.ModuleList] = None
        self.output_feats_adapter: Optional[nn.Linear] = None
        self._infer_backbone_layers_indices()
        self._make_fuser(feat_fuser)
        self._make_adapters()

    def has_param_groups(self) -> bool:
        return (
            super().has_param_groups()
            or self.backbone_weight_decay is not None
            or self.backbone_lr is not None
            or self.backbone_feats_weight_decay is not None
            or self.backbone_feats_lr is not None
            or self.adapter_weight_decay is not None
            or self.adapter_lr is not None
        )

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        if (
            self.backbone_weight_decay is None
            and self.backbone_lr is None
            and self.backbone_feats_weight_decay is None
            and self.backbone_feats_lr is None
            and self.adapter_weight_decay is None
            and self.adapter_lr is None
        ):
            return super().trainable_param_groups()

        backbone = []
        backbone_feat_extractor = []
        bias = []
        backbone_feats_extractor_bias = []
        backbone_bias = []
        if self.bias_weight_decay is None:
            backbone_feat_extractor = list(self.hf_feats.feat_extract_params())
            backbone = list(self.hf_feats.trainable_encoder_params())
        else:
            backbone_feat_extractor = list(
                self.hf_feats.feat_extract_params(bias=False)
            )
            backbone = list(self.hf_feats.trainable_encoder_params(bias=False))
            if self.backbone_feats_lr is None:
                bias += list(self.hf_feats.feat_extract_bias())
            else:
                backbone_feats_extractor_bias += list(self.hf_feats.feat_extract_bias())

            if self.backbone_lr is None:
                bias += list(self.hf_feats.encoder_bias())
            else:
                backbone_bias += list(self.hf_feats.encoder_bias())

        adapters = []
        qformer = []
        proj_head = []
        head = []
        other = []
        for name, param in self.trainable_named_parameters():
            # we do not regularize biases nor Norm parameters
            if name.startswith("hf_feats"):
                continue

            if self.bias_weight_decay is not None and (
                name.endswith(".bias") or len(param.shape) == 1
            ):
                bias.append(param)
            else:
                if name.startswith("hidden_feats_adapter") or name.startswith(
                    "output_feats_adapter"
                ):
                    adapters.append(param)
                elif self.qformer_weight_decay is not None and (
                    name.startswith("hidden_feats_agg_qformer")
                    or name.startswith("output_feats_agg_qformer")
                ):
                    qformer.append(param)
                elif self.proj_head_weight_decay is not None and name.startswith(
                    "proj_head"
                ):
                    proj_head.append(param)
                elif self.head_weight_decay is not None and name.startswith("head"):
                    head.append(param)
                else:
                    other.append(param)

        trainable_params = []
        if backbone:
            backbone_params = {"params": backbone}
            if self.backbone_lr is not None:
                backbone_params["lr"] = self.backbone_lr
            if self.backbone_weight_decay is not None:
                backbone_params["weight_decay"] = self.backbone_weight_decay

            trainable_params.append(backbone_params)

        if backbone_feat_extractor:
            backbone_feat_extractor_params = {"params": backbone_feat_extractor}
            if self.backbone_feats_lr is not None:
                backbone_feat_extractor_params["lr"] = self.backbone_feats_lr
            if self.backbone_feats_weight_decay is not None:
                backbone_feat_extractor_params["weight_decay"] = (
                    self.backbone_feats_weight_decay
                )

            trainable_params.append(backbone_feat_extractor_params)

        if backbone_bias:
            backbone_bias_params = {"params": backbone_bias}
            if self.backbone_lr is not None:
                backbone_bias_params["lr"] = self.backbone_lr
            if self.bias_weight_decay is not None:
                backbone_bias_params["weight_decay"] = self.bias_weight_decay

            trainable_params.append(backbone_bias_params)

        if backbone_feats_extractor_bias:
            backbone_feats_extractor_bias_params = {
                "params": backbone_feats_extractor_bias
            }
            if self.backbone_feats_lr is not None:
                backbone_feats_extractor_bias_params["lr"] = self.backbone_feats_lr
            if self.bias_weight_decay is not None:
                backbone_feats_extractor_bias_params["weight_decay"] = (
                    self.bias_weight_decay
                )

            trainable_params.append(backbone_feats_extractor_bias_params)

        if adapters:
            adapter_params = {"params": adapters}
            if self.adapter_lr is not None:
                adapter_params["lr"] = self.adapter_lr
            if self.adapter_weight_decay is not None:
                adapter_params["weight_decay"] = self.adapter_weight_decay

            trainable_params.append(adapter_params)

        if other:
            trainable_params.append({"params": other})
        if qformer:
            trainable_params.append(
                {"params": qformer, "weight_decay": self.qformer_weight_decay}
            )
        if proj_head:
            trainable_params.append(
                {"params": proj_head, "weight_decay": self.proj_head_weight_decay}
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

    @property
    def sample_frequency(self) -> int:
        """int: Sampling frequency assumed by ``hf_feats``."""
        return self.hf_feats.sample_frequency

    def _make_adapters(self) -> None:
        """Build linear adapters that map backbone tensors to Q-former inputs."""
        self.hidden_feats_adapter = None
        self.output_feats_adapter = None
        in_feats = self.hf_feats.hidden_size
        if self.hidden_feats_agg_qformer is not None:
            hfa_qformer_in_feats = self.hidden_feats_agg_qformer.in_feats
            if self.hidden_feats_shared_adapters:
                if hfa_qformer_in_feats != in_feats:
                    adapter = nn.Linear(in_feats, hfa_qformer_in_feats)
                    self.hidden_feats_adapter = nn.ModuleList(
                        [adapter]
                        * (
                            self.hf_feats.num_encoder_layers
                            - self.hidden_feats_agg_start
                        )
                    )
                else:
                    self.hidden_feats_adapter = None
            else:
                self.hidden_feats_adapter = nn.ModuleList(
                    [nn.Linear(in_feats, hfa_qformer_in_feats)]
                    * (
                        self.hf_feats.num_encoder_layers
                        + 1
                        - self.hidden_feats_agg_start
                    )
                )
        else:
            self.hidden_feats_adapter = None

    def _make_fuser(self, feat_fuser: Optional[Dict[str, Any]]) -> None:
        """Builds the feature-fusion module based on HF extractor dimensions.

        Args:
          feat_fuser: Optional configuration dictionary for ``FeatFuserMVN``.

        Returns:
          None.
        """
        if self.output_feats_agg_qformer is None:
            self.feat_fuser = None
            return
        if feat_fuser is None:
            raise ValueError(
                "feat_fuser must be provided when output_feats_agg_qformer is not None"
            )

        num_feats = (
            self.hf_feats.num_encoder_layers + 1 - self.hidden_feats_fusion_start
        )
        feat_dim = self.hf_feats.hidden_size
        feat_fuser["feat_fuser"]["num_feats"] = num_feats
        feat_fuser["feat_fuser"]["feat_dim"] = feat_dim
        ofa_qformer_in_feats = self.output_feats_agg_qformer.in_feats
        if ofa_qformer_in_feats != feat_dim:
            feat_fuser["feat_fuser"]["proj_dim"] = ofa_qformer_in_feats
            feat_fuser["feat_fuser"]["proj_bias"] = False
        else:
            feat_fuser["feat_fuser"]["proj_dim"] = None
            feat_fuser["feat_fuser"]["proj_bias"] = False
        self.feat_fuser = FeatFuserMVN(**feat_fuser)

    def freeze_backbone(self) -> None:
        self.hf_feats.freeze()

    def freeze_backbone_feat_extractor(self) -> None:
        self.hf_feats.freeze_feature_encoder()

    def freeze_adapters(self) -> None:
        if self.feat_fuser is not None:
            for param in self.feat_fuser.parameters():
                param.requires_grad = False

        if self.hidden_feats_adapter is not None:
            for adapter in self.hidden_feats_adapter:
                for param in adapter.parameters():
                    param.requires_grad = False

        if self.output_feats_adapter is not None:
            for param in self.output_feats_adapter.parameters():
                param.requires_grad = False

    def set_backbone_in_eval_mode(self) -> None:
        self.hf_feats.eval()

    def set_backbone_in_train_mode(self) -> None:
        self.hf_feats.train()

    def set_adapters_in_train_mode(self) -> None:
        pass

    def set_adapters_in_eval_mode(self) -> None:
        pass

    def _infer_backbone_layers_indices(self):
        pass

    def change_config(
        self,
        backbone_feats_lr: Optional[float] = None,
        backbone_feats_weight_decay: Optional[float] = None,
        backbone_lr: Optional[float] = None,
        backbone_weight_decay: Optional[float] = None,
        adapter_lr: Optional[float] = None,
        adapter_weight_decay: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Change model configuration at runtime.

        Args:
            backbone_feats_lr: Optional learning-rate override for front-end
                feature/backbone parameters.
            backbone_feats_weight_decay: Optional weight-decay override for
                front-end feature/backbone parameters.
            backbone_lr: Optional learning-rate override for fused-backbone
                parameters.
            backbone_weight_decay: Optional weight-decay override for
                fused-backbone parameters.
            adapter_lr: Optional learning-rate override for adapter parameters
                (``hidden_feats_adapter`` and ``output_feats_adapter``).
            adapter_weight_decay: Optional weight-decay override for adapter
                parameters (``hidden_feats_adapter`` and ``output_feats_adapter``).
            **kwargs: Additional keyword arguments forwarded to the base class
                method for reconfiguration.
        """

        if backbone_feats_lr is not None:
            logging.info(
                "overriding backbone-feats learning rate with new value: %s",
                backbone_feats_lr,
            )
            self.backbone_feats_lr = backbone_feats_lr

        if backbone_feats_weight_decay is not None:
            logging.info(
                "overriding backbone-feats weight decay with new value: %s",
                backbone_feats_weight_decay,
            )
            self.backbone_feats_weight_decay = backbone_feats_weight_decay

        if backbone_lr is not None:
            logging.info(
                "overriding backbone learning rate with new value: %s", backbone_lr
            )
            self.backbone_lr = backbone_lr

        if backbone_weight_decay is not None:
            logging.info(
                "overriding backbone weight decay with new value: %s",
                backbone_weight_decay,
            )
            self.backbone_weight_decay = backbone_weight_decay

        if adapter_lr is not None:
            logging.info(
                "overriding adapter learning rate with new value: %s", adapter_lr
            )
            self.adapter_lr = adapter_lr

        if adapter_weight_decay is not None:
            logging.info(
                "overriding adapter weight decay with new value: %s",
                adapter_weight_decay,
            )
            self.adapter_weight_decay = adapter_weight_decay

        super().change_config(**kwargs)

    def forward_backbone(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_hidden_feats: bool = False,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Run HF feature extractor/backbone.

        Args:
            x: Input waveform tensor shaped ``(batch, samples)``.
            x_lengths: Optional lengths per input example.
            return_hidden_feats: Whether to also return hidden layer activations.

        Returns:
            Tuple with backbone outputs, their lengths, hidden features, and hidden
            feature lengths (entries are ``None`` when unavailable).
        """
        return_hid_states = (
            False
            if not return_hidden_feats
            and (self.feat_fuser is None or self.feat_fuser.fuser_type == "last")
            else True
        )
        with self._hf_context:
            hf_output = self.hf_feats(
                x,
                x_lengths,
                return_hid_states=return_hid_states,
            )

        feat_lengths = hf_output["hidden_states_lengths"]
        if return_hid_states:
            hid_feats = hf_output["hidden_states"]
        else:
            hid_feats = [hf_output["last_hidden_state"]]

        if isinstance(hid_feats, tuple):
            hid_feats = list(hid_feats)

        if self.backbone_return_output:
            if return_hid_states:
                backbone_feats = hid_feats[self.hidden_feats_fusion_start :]
            else:
                backbone_feats = hid_feats

            backbone_feats_lengths = feat_lengths

        else:
            backbone_feats, backbone_feats_lengths = None, None

        if return_hidden_feats:
            backbone_hidden_feats = hid_feats[self.hidden_feats_agg_start :]
            backbone_hidden_feats_lengths = feat_lengths
        else:
            backbone_hidden_feats = None
            backbone_hidden_feats_lengths = None

        return (
            backbone_feats,
            backbone_feats_lengths,
            backbone_hidden_feats,
            backbone_hidden_feats_lengths,
        )

    def forward_adapter(
        self,
        backbone_output_feats: Optional[torch.Tensor] = None,
        backbone_output_feats_lengths: Optional[torch.Tensor] = None,
        backbone_hidden_feats: Optional[List[torch.Tensor]] = None,
        backbone_hidden_feats_lengths: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Project backbone tensors into the Q-former input spaces.

        Args:
            backbone_output_feats: Output features returned by the backbone.
            backbone_output_feats_lengths: Optional lengths of the output features.
            backbone_hidden_feats: Optional list of hidden backbone activations.
            backbone_hidden_feats_lengths: Optional lengths of the hidden activations.

        Returns:
            Tuple mirroring the inputs but with tensors mapped through adapters so
            they match each Q-former input dimension.
        """
        if self.hidden_feats_adapter is not None:
            assert backbone_hidden_feats is not None
            adapted_hidden_feats = []
            for h, adapter in zip(backbone_hidden_feats, self.hidden_feats_adapter):
                h_adapted = adapter(h)
                adapted_hidden_feats.append(h_adapted)
        else:
            adapted_hidden_feats = backbone_hidden_feats

        if self.feat_fuser is not None:
            adapted_output_feats, adapted_output_feats_lengths = self.feat_fuser(
                backbone_output_feats, backbone_output_feats_lengths
            )
        elif backbone_output_feats is None:
            adapted_output_feats = None
            adapted_output_feats_lengths = backbone_output_feats_lengths
        else:
            adapted_output_feats = backbone_output_feats[-1]
            adapted_output_feats_lengths = backbone_output_feats_lengths

        return (
            adapted_output_feats,
            adapted_output_feats_lengths,
            adapted_hidden_feats,
            backbone_hidden_feats_lengths,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable dictionary capturing constructor arguments.

        Returns:
            Dict[str, Any]: Configuration for acoustic features, backbone, and base
            ``QVector`` options.
        """
        hf_cfg = self.hf_feats.get_config(no_class_name=True)
        fuser_cfg = (
            None
            if self.feat_fuser is None
            else self.feat_fuser.get_config(no_class_name=True)
        )
        base_config = super().get_config()
        config = {
            "hf_feats": hf_cfg,
            "feat_fuser": fuser_cfg,
            "hidden_feats_fusion_start": self.hidden_feats_fusion_start,
            "hidden_feats_agg_start": self.hidden_feats_agg_start,
            "hidden_feats_shared_adapters": self.hidden_feats_shared_adapters,
            "backbone_feats_lr": self.backbone_feats_lr,
            "backbone_feats_weight_decay": self.backbone_feats_weight_decay,
            "backbone_lr": self.backbone_lr,
            "backbone_weight_decay": self.backbone_weight_decay,
            "adapter_lr": self.adapter_lr,
            "adapter_weight_decay": self.adapter_weight_decay,
        }
        config.update(base_config)
        return config

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Return only keyword args that match the constructor signature."""
        return filter_func_args(HFWav2QVector.__init__, kwargs)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register CLI/configuration arguments for this model.

        Args:
            parser: ``ArgumentParser`` that receives the class arguments.
            prefix: Optional namespace prefix for grouped argument registration.
            skip: Optional set of argument names to exclude from registration.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        skip = set(skip) if skip is not None else set()

        if "feat_fuser" not in skip:
            FeatFuserMVN.add_class_args(parser, prefix="feat_fuser")
        if "hidden_feats_fusion_start" not in skip:
            parser.add_argument(
                "--hidden-feats-fusion-start",
                type=int,
                default=0,
                help="first hidden-state layer index used for output-feature fusion",
            )
        if "hidden_feats_agg_start" not in skip:
            parser.add_argument(
                "--hidden-feats-agg-start",
                type=int,
                default=0,
                help="first hidden-state layer index used for hidden-feature aggregation",
            )
        if "hidden_feats_shared_adapters" not in skip:
            parser.add_argument(
                "--hidden-feats-shared-adapters",
                default=True,
                action=ActionYesNo,
                help="share hidden-feature adapter weights across selected layers",
            )
        if "backbone_feats_lr" not in skip:
            parser.add_argument(
                "--backbone-feats-lr",
                type=float,
                default=None,
                help="optional learning-rate override for front-end feature/backbone parameters",
            )
        if "backbone_feats_weight_decay" not in skip:
            parser.add_argument(
                "--backbone-feats-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for front-end feature/backbone parameters",
            )
        if "backbone_lr" not in skip:
            parser.add_argument(
                "--backbone-lr",
                type=float,
                default=None,
                help="optional learning-rate override for fused-backbone parameters",
            )
        if "backbone_weight_decay" not in skip:
            parser.add_argument(
                "--backbone-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for fused-backbone parameters",
            )
        if "adapter_lr" not in skip:
            parser.add_argument(
                "--adapter-lr",
                type=float,
                default=None,
                help="optional learning-rate override for adapter parameters",
            )
        if "adapter_weight_decay" not in skip:
            parser.add_argument(
                "--adapter-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for adapter parameters",
            )
        QVector.add_class_args(parser, skip=skip)

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        base_args = QVector.filter_finetune_args(**kwargs)
        child_args = filter_func_args(HFWav2QVector.change_config, kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[Set[str]] = None,
    ) -> None:
        """Register fine-tuning CLI/configuration arguments for this model.

        Args:
            parser: ``ArgumentParser`` that receives fine-tuning arguments.
            prefix: Optional namespace prefix for grouped argument registration.
            skip: Optional set of argument names to exclude from registration.
        """
        if skip is None:
            skip = set()
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        if "backbone_feats_lr" not in skip:
            parser.add_argument(
                "--backbone-feats-lr",
                type=float,
                default=None,
                help="optional learning-rate override for front-end feature/backbone parameters",
            )
        if "backbone_feats_weight_decay" not in skip:
            parser.add_argument(
                "--backbone-feats-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for front-end feature/backbone parameters",
            )
        if "backbone_lr" not in skip:
            parser.add_argument(
                "--backbone-lr",
                type=float,
                default=None,
                help="optional learning-rate override for fused-backbone parameters",
            )
        if "backbone_weight_decay" not in skip:
            parser.add_argument(
                "--backbone-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for fused-backbone parameters",
            )
        if "adapter_lr" not in skip:
            parser.add_argument(
                "--adapter-lr",
                type=float,
                default=None,
                help="optional learning-rate override for adapter parameters",
            )
        if "adapter_weight_decay" not in skip:
            parser.add_argument(
                "--adapter-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for adapter parameters",
            )
        QVector.add_finetune_args(parser, skip=skip)

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

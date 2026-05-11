"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import filter_func_args
from ...narchs import AudioFeatsMVN, HydraHead, QFormerV2, ResNet
from ...narchs import ResNetFactory as RNF
from ...utils.masking import scale_seq_lengths
from ..wav2xvectors import Wav2ResNetXVector as RXVec
from .qvector import QVector


class ResNetQVector(QVector):
    """Q-vector model that combines a ResNet encoder with optional adapters.

    Attributes:
        acoustic_feats: Feature-extraction front-end that converts raw waveforms
            into acoustic representations.
        resnet_encoder: ResNet backbone that processes acoustic features.
        resnet_type: Identifier for the instantiated backbone variant.
        backbone_layers: Indices of intermediate backbone layers returned during
            hidden-feature aggregation, or ``None`` when not requested.
        backbone_return_output: Flag indicating whether ``forward_hid_feats`` also
            returns the final backbone output tensor.
        hidden_feats_adapter: Optional projection layers that align backbone hidden
            features with the hidden Q-former input dimensionality.
        output_feats_adapter: Optional projection mapping backbone outputs to the
            output Q-former input space.
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
        acoustic_feats: Union[Dict[str, Any], AudioFeatsMVN],
        resnet_encoder: Dict[str, Any],
        hidden_feats_agg_qformer: Union[Dict[str, Any], QFormerV2, None],
        num_hidden_feats_queries: int,
        output_feats_agg_qformer: Union[Dict[str, Any], None],
        num_output_feats_queries: int,
        qvector_dim: int,
        head: Union[Dict[str, Any], HydraHead],
        resnet_lr: Optional[float] = None,
        resnet_weight_decay: Optional[float] = None,
        adapter_lr: Optional[float] = None,
        adapter_weight_decay: Optional[float] = None,
        qformer_weight_decay: Optional[float] = None,
        proj_head_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialise the ResNet-backed q-vector model.

        Args:
            acoustic_feats: Acoustic feature extractor configuration or instance.
            resnet_encoder: Keyword arguments for :class:`ResNetFactory`.
            hidden_feats_agg_qformer: Hidden Q-former configuration or module.
            num_hidden_feats_queries: Number of hidden queries used for aggregation.
            output_feats_agg_qformer: Output Q-former configuration or module.
            num_output_feats_queries: Number of output queries.
            qvector_dim: Size of the final q-vector embedding.
            head: Hydra head configuration or module.
            resnet_lr: Optional learning-rate override for backbone
                ``resnet_encoder`` parameters.
            resnet_weight_decay: Optional weight-decay override for backbone
                ``resnet_encoder`` parameters.
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
        if isinstance(acoustic_feats, dict):
            logging.info("making acoustic feature extractor")
            acoustic_feats = AudioFeatsMVN.filter_args(**acoustic_feats)
            acoustic_feats["trans"] = True
            acoustic_feats = AudioFeatsMVN(**acoustic_feats)
        else:
            assert isinstance(acoustic_feats, AudioFeatsMVN)

        assert isinstance(resnet_encoder, dict)
        resnet_type = resnet_encoder["resnet_type"]
        logging.info("making %s encoder network", resnet_type)
        resnet_encoder["in_feats"] = acoustic_feats.out_feats
        resnet_encoder = RNF.filter_args(**resnet_encoder)
        resnet_encoder = RNF.create(**resnet_encoder)

        super().__init__(
            hidden_feats_agg_qformer=hidden_feats_agg_qformer,
            num_hidden_feats_queries=num_hidden_feats_queries,
            output_feats_agg_qformer=output_feats_agg_qformer,
            num_output_feats_queries=num_output_feats_queries,
            qvector_dim=qvector_dim,
            head=head,
            qformer_weight_decay=qformer_weight_decay,
            proj_head_weight_decay=proj_head_weight_decay,
            head_weight_decay=head_weight_decay,
            bias_weight_decay=bias_weight_decay,
        )
        self.acoustic_feats: AudioFeatsMVN = acoustic_feats
        self.resnet_encoder: ResNet = resnet_encoder
        self.resnet_type: str = resnet_type
        self.resnet_lr = resnet_lr
        self.resnet_weight_decay = resnet_weight_decay
        self.adapter_lr = adapter_lr
        self.adapter_weight_decay = adapter_weight_decay
        self._acoustic_feats_context = torch.no_grad()
        self.backbone_layers: Optional[List[int]] = None
        self.backbone_return_output: bool = False
        self.hidden_feats_adapter: Optional[nn.ModuleList] = None
        self.output_feats_adapter: Optional[nn.Linear] = None
        self._infer_backbone_layer_indices()
        self._make_adapters()

    def has_param_groups(self):
        return (
            super().has_param_groups()
            or self.resnet_weight_decay is not None
            or self.resnet_lr is not None
            or self.adapter_weight_decay is not None
            or self.adapter_lr is not None
        )

    def trainable_param_groups(self):
        if (
            self.resnet_weight_decay is None
            and self.resnet_lr is None
            and self.adapter_weight_decay is None
            and self.adapter_lr is None
        ):
            return super().trainable_param_groups()

        resnet = []
        adapters = []
        qformer = []
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
                if name.startswith("resnet_encoder"):
                    resnet.append(param)
                elif name.startswith("hidden_feats_adapter") or name.startswith(
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
        if resnet:
            resnet_params = {"params": resnet}
            if self.resnet_lr is not None:
                resnet_params["lr"] = self.resnet_lr
            if self.resnet_weight_decay is not None:
                resnet_params["weight_decay"] = self.resnet_weight_decay

            trainable_params.append(resnet_params)

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
        """int: Sampling frequency assumed by ``acoustic_feats``."""
        return self.acoustic_feats.sample_frequency

    @property
    def max_chunk_length(self) -> int:
        """Maximum chunk length (in samples) seen during training."""
        return 0

    def _infer_backbone_layer_indices(self) -> None:
        """Determine which backbone layers to capture for aggregation."""
        if self.output_feats_agg_qformer is None:
            self.backbone_layers = [1, 2, 3, 4]
            self.backbone_return_output = False
        elif self.hidden_feats_agg_qformer is None:
            self.backbone_layers = None
            self.backbone_return_output = True
        else:
            self.backbone_layers = [1, 2, 3]
            self.backbone_return_output = True

    def _make_adapters(self) -> None:
        """Build linear adapters that map backbone tensors to Q-former inputs."""
        self.hidden_feats_adapter = None
        self.output_feats_adapter = None
        in_feats = self.acoustic_feats.out_feats
        in_shape = (1, 1, in_feats, None)
        if self.hidden_feats_agg_qformer is not None:
            hfa_qformer_in_feats = self.hidden_feats_agg_qformer.in_feats
            hid_shapes = self.resnet_encoder.hid_shapes(
                in_shape=in_shape, layers=self.backbone_layers
            )
            hid_feats = [s[1] * s[2] for s in hid_shapes]
            self.hidden_feats_adapter = nn.ModuleList(
                [nn.Linear(hf, hfa_qformer_in_feats) for hf in hid_feats]
            )
        else:
            self.hidden_feats_adapter = None

        if self.output_feats_agg_qformer is not None:
            ofa_qformer_in_feats = self.output_feats_agg_qformer.in_feats
            out_shape = self.resnet_encoder.out_shape(in_shape)
            out_feats = out_shape[1] * out_shape[2]
            if out_feats != ofa_qformer_in_feats:
                self.output_feats_adapter = nn.Linear(out_feats, ofa_qformer_in_feats)
            else:
                self.output_feats_adapter = None
        else:
            self.output_feats_adapter = None

    def init_from_xvector(self, xvector_model: RXVec):
        """Initialize q-vector model backbone parameters from a pre-trained x-vector model.

        Args:
            xvector_model: Pre-trained x-vector model to use for initialization.
        """
        assert isinstance(xvector_model, RXVec)
        feats = xvector_model.feats
        feats.spec_augment = self.acoustic_feats.spec_augment
        self.acoustic_feats = feats
        self.resnet_encoder.load_state_dict(
            xvector_model.xvector.encoder_net.state_dict()
        )

    def freeze_backbone(self):
        self.resnet_encoder.freeze()

    def set_backbone_in_eval_mode(self):
        self.resnet_encoder.eval()

    def set_adapters_in_train_mode(self):
        pass

    def change_config(
        self,
        encoder_dropout_rate: Optional[float] = None,
        resnet_lr: Optional[float] = None,
        resnet_weight_decay: Optional[float] = None,
        adapter_lr: Optional[float] = None,
        adapter_weight_decay: Optional[float] = None,
        **kwargs,
    ):
        """Change model configuration at runtime.

        Args:
            encoder_dropout_rate: Optional dropout rate to apply in the ResNet
                encoder during fine-tuning.
            resnet_lr: Optional learning-rate override for backbone
                ``resnet_encoder`` parameters.
            resnet_weight_decay: Optional weight-decay override for backbone
                ``resnet_encoder`` parameters.
            adapter_lr: Optional learning-rate override for adapter parameters
                (``hidden_feats_adapter`` and ``output_feats_adapter``).
            adapter_weight_decay: Optional weight-decay override for adapter
                parameters (``hidden_feats_adapter`` and ``output_feats_adapter``).
            **kwargs: Additional keyword arguments forwarded to the base class
                method for reconfiguration.
        """

        if encoder_dropout_rate is not None:
            self.resnet_encoder.change_dropouts(dropout_rate=encoder_dropout_rate)

        if resnet_lr is not None:
            logging.info(
                "overriding resnet learning rate with new value: %s", resnet_lr
            )
            self.resnet_lr = resnet_lr

        if resnet_weight_decay is not None:
            logging.info(
                "overriding resnet weight decay with new value: %s",
                resnet_weight_decay,
            )
            self.resnet_weight_decay = resnet_weight_decay

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
        """Run acoustic front-end and resnet backbone.

        Args:
            x: Input waveform tensor shaped ``(batch, channels, samples)``.
            x_lengths: Optional lengths per input example.
            return_hidden_feats: Whether to also return hidden layer activations.

        Returns:
            Tuple with backbone outputs, their lengths, hidden features, and hidden
            feature lengths (entries are ``None`` when unavailable).
        """
        with self._acoustic_feats_context:
            x, x_lengths = self.acoustic_feats(x, x_lengths)
            x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
            max_in_length = x.size(3)

        if return_hidden_feats:
            backbone_hidden_feats = self.resnet_encoder.forward_hid_feats(
                x,
                x_lengths,
                layers=self.backbone_layers,
                return_output=self.backbone_return_output,
            )
            if self.backbone_return_output:
                backbone_hidden_feats, backbone_feats = backbone_hidden_feats
            else:
                backbone_feats = None
        else:
            backbone_feats = self.resnet_encoder(x, x_lengths)
            backbone_hidden_feats = None

        if backbone_feats is not None:
            backbone_feats = backbone_feats.view(
                backbone_feats.size(0), -1, backbone_feats.size(3)
            ).transpose(1, 2)
            backbone_feats_lengths = scale_seq_lengths(
                x_lengths,
                backbone_feats.size(1),
                max_in_length,
            )
        else:
            backbone_feats_lengths = None

        if return_hidden_feats:
            backbone_hidden_feats = [
                h.view(h.size(0), -1, h.size(3)).transpose(1, 2)
                for h in backbone_hidden_feats
            ]
            backbone_hidden_feats_lengths = [
                scale_seq_lengths(x_lengths, h.size(1), max_in_length)
                for h in backbone_hidden_feats
            ]
            return (
                backbone_feats,
                backbone_feats_lengths,
                backbone_hidden_feats,
                backbone_hidden_feats_lengths,
            )
        else:
            return backbone_feats, backbone_feats_lengths, None, None

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

        if self.output_feats_adapter is not None:
            assert backbone_output_feats is not None
            adapted_output_feats = self.output_feats_adapter(backbone_output_feats)
        else:
            adapted_output_feats = backbone_output_feats

        return (
            adapted_output_feats,
            backbone_output_feats_lengths,
            adapted_hidden_feats,
            backbone_hidden_feats_lengths,
        )

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable dictionary capturing constructor arguments.

        Returns:
            Dict[str, Any]: Configuration for acoustic features, backbone, and base
            ``QVector`` options.
        """
        feats_cfg = self.acoustic_feats.get_config(no_class_name=True)
        resnet_cfg = {
            "resnet_type": self.resnet_type,
            "in_channels": self.resnet_encoder.in_channels,
            "conv_channels": self.resnet_encoder.conv_channels,
            "base_channels": self.resnet_encoder.base_channels,
            "hid_act": self.resnet_encoder.hid_act,
            "in_kernel_size": self.resnet_encoder.in_kernel_size,
            "in_stride": self.resnet_encoder.in_stride,
            "zero_init_residual": self.resnet_encoder.zero_init_residual,
            "groups": self.resnet_encoder.groups,
            "replace_stride_with_dilation": self.resnet_encoder.replace_stride_with_dilation,
            "dropout_rate": self.resnet_encoder.dropout_rate,
            "norm_layer": self.resnet_encoder.norm_layer,
            "norm_before": self.resnet_encoder.norm_before,
            "do_maxpool": self.resnet_encoder.do_maxpool,
            "in_norm": self.resnet_encoder.in_norm,
            "se_r": self.resnet_encoder.se_r,
            "res2net_scale": self.resnet_encoder.res2net_scale,
            "res2net_width_factor": self.resnet_encoder.res2net_width_factor,
            "freq_pos_enc": self.resnet_encoder.freq_pos_enc,
        }
        base_config = super().get_config()
        config = {
            "acoustic_feats": feats_cfg,
            "resnet_encoder": resnet_cfg,
            "resnet_lr": self.resnet_lr,
            "resnet_weight_decay": self.resnet_weight_decay,
            "adapter_lr": self.adapter_lr,
            "adapter_weight_decay": self.adapter_weight_decay,
        }
        config.update(base_config)
        return config

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "ResNetQVector":
        """Instantiate a model from serialized configuration/state.

        Args:
            file_path: Optional path to a checkpoint bundle.
            cfg: Optional configuration dictionary to override disk contents.
            state_dict: Optional PyTorch state dictionary.

        Returns:
            ResNetQVector: Model with configuration/state restored.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Return only keyword args that match the constructor signature."""
        return filter_func_args(ResNetQVector.__init__, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        """Register CLI/configuration arguments for this model.

        Args:
            parser: ``ArgumentParser`` that receives the class arguments.
            prefix: Optional namespace prefix for grouped argument registration.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AudioFeatsMVN.add_class_args(parser, prefix="acoustic_feats")
        RNF.add_class_args(parser, prefix="resnet_encoder")
        parser.add_argument(
            "--resnet-lr",
            type=float,
            default=None,
            help="optional learning-rate override for backbone resnet_encoder parameters",
        )
        parser.add_argument(
            "--resnet-weight-decay",
            type=float,
            default=None,
            help="optional weight-decay override for backbone resnet_encoder parameters",
        )
        parser.add_argument(
            "--adapter-lr",
            type=float,
            default=None,
            help="optional learning-rate override for adapter parameters",
        )
        parser.add_argument(
            "--adapter-weight-decay",
            type=float,
            default=None,
            help="optional weight-decay override for adapter parameters",
        )
        QVector.add_class_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs):
        base_args = QVector.filter_finetune_args(**kwargs)
        child_args = filter_func_args(ResNetQVector.change_config, kwargs)
        base_args.update(child_args)
        return base_args

    @staticmethod
    def add_finetune_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--encoder-dropout-rate",
            type=float,
            default=None,
            help="Optional dropout rate to apply in the ResNet encoder during fine-tuning.",
        )
        parser.add_argument(
            "--resnet-lr",
            type=float,
            default=None,
            help="optional learning-rate override for backbone resnet_encoder parameters",
        )
        parser.add_argument(
            "--resnet-weight-decay",
            type=float,
            default=None,
            help="optional weight-decay override for backbone resnet_encoder parameters",
        )
        parser.add_argument(
            "--adapter-lr",
            type=float,
            default=None,
            help="optional learning-rate override for adapter parameters",
        )
        parser.add_argument(
            "--adapter-weight-decay",
            type=float,
            default=None,
            help="optional weight-decay override for adapter parameters",
        )
        QVector.add_finetune_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

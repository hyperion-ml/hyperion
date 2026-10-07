"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ....utils.misc import PathLike, filter_func_args
from ...hyper_torch_model import HyperTorchModel
from ...narchs import AudioFeatsMVN, HydraHead, ResNet
from ...narchs import ResNetFactory as RNF
from ...utils.masking import scale_seq_lengths
from ..wav2xvectors import Wav2ResNetXVector as RXVec
from .xvectorp import XVectorP


class ResNetXVectorP(XVectorP):
    """Waveform x-vector+ model with acoustic features and a ResNet backbone.

    Attributes:
        acoustic_feats: Acoustic feature extractor for waveform input.
        resnet_encoder: ResNet producing frame-level features.
        resnet_type: ResNet factory identifier.
        backbone_layers: Intermediate stages returned for feature inspection.
        resnet_lr: Optional backbone learning rate.
        resnet_weight_decay: Optional backbone weight decay.
        pooling: Global pooling module (inherited).
        proj_head: Projection with optional input or output normalization (inherited).
        head: Optional classification or regression head (inherited).
        xvector_dim: Embedding dimension (inherited).
        train_mode: Active training regime (inherited).
    """

    def __init__(
        self,
        acoustic_feats: Union[Dict[str, Any], AudioFeatsMVN],
        resnet_encoder: Dict[str, Any],
        pooling: Union[str, Dict[str, Any], nn.Module],
        xvector_dim: int,
        head: Optional[Union[Dict[str, Any], HydraHead]],
        proj_norm_layer: Optional[str] = None,
        proj_use_norm: bool = True,
        proj_norm_before: bool = True,
        enable_xvector_sig_reg: bool = False,
        xvector_sig_reg: Optional[Dict[str, Any]] = None,
        resnet_lr: Optional[float] = None,
        resnet_weight_decay: Optional[float] = None,
        pooling_weight_decay: Optional[float] = None,
        proj_weight_decay: Optional[float] = None,
        head_weight_decay: Optional[float] = None,
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        """Initialise the ResNet-backed x-vector model.

        Args:
            acoustic_feats: Acoustic feature extractor configuration or instance.
            resnet_encoder: Keyword arguments for :class:`ResNetFactory`.
            pooling: Global pooling configuration or module.
            xvector_dim: Size of the final x-vector embedding.
            head: Hydra head configuration or module, or ``None`` for embedding-only use.
            proj_norm_layer: Batch, layer, or RMS normalization type.
            proj_use_norm: Whether projection normalization is enabled.
            proj_norm_before: Whether normalization precedes projection.

            resnet_lr: Optional learning-rate override for backbone
                ``resnet_encoder`` parameters.
            resnet_weight_decay: Optional weight-decay override for backbone
                ``resnet_encoder`` parameters.
            enable_xvector_sig_reg: Whether to calculate SIGReg on projected embeddings.
            xvector_sig_reg: SIGReg constructor arguments.
            pooling_weight_decay: Optional pooling weight decay.
            proj_weight_decay: Optional weight-decay override applied to
                projection-head parameters.
            head_weight_decay: Optional weight-decay override applied to downstream
                head parameters.
            bias_weight_decay: Optional weight decay for biases and normalization parameters.
        """
        if isinstance(acoustic_feats, dict):
            logging.info("making acoustic feature extractor")
            acoustic_feats = AudioFeatsMVN.filter_args(**acoustic_feats)
            acoustic_feats["trans"] = True
            acoustic_feats = AudioFeatsMVN(**acoustic_feats)
        elif not isinstance(acoustic_feats, AudioFeatsMVN):
            raise TypeError("acoustic_feats must be an AudioFeatsMVN module or config")

        if not isinstance(resnet_encoder, dict):
            raise TypeError("resnet_encoder must be a configuration dictionary")
        resnet_encoder = dict(resnet_encoder)
        resnet_type = resnet_encoder["resnet_type"]
        logging.info("making %s encoder network", resnet_type)
        resnet_encoder["in_feats"] = acoustic_feats.out_feats
        resnet_encoder = RNF.filter_args(**resnet_encoder)
        resnet_encoder = RNF.create(**resnet_encoder)

        # Feature width is needed while the base class builds pooling/projection.
        out_shape = resnet_encoder.out_shape((None, 1, acoustic_feats.out_feats, None))
        self._backbone_output_feats = out_shape[1] * out_shape[2]
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
        self.acoustic_feats: AudioFeatsMVN = acoustic_feats
        self.resnet_encoder: ResNet = resnet_encoder
        self.resnet_type: str = resnet_type
        self.resnet_lr = resnet_lr
        self.resnet_weight_decay = resnet_weight_decay
        self._acoustic_feats_context = torch.no_grad()
        self.backbone_layers: Optional[List[int]] = None
        self._infer_backbone_layers_indices()

    def has_param_groups(self) -> bool:
        """Return whether the model exposes custom optimizer parameter groups.

        Returns:
            ``True`` when custom parameter groups are configured.
        """
        return (
            super().has_param_groups()
            or self.resnet_weight_decay is not None
            or self.resnet_lr is not None
        )

    def trainable_param_groups(self) -> List[Dict[str, Any]]:
        """Return optimizer parameter groups for the trainable components.

        Returns:
            Parameter groups with optional component-specific weight decay.
        """
        if self.resnet_weight_decay is None and self.resnet_lr is None:
            return super().trainable_param_groups()

        resnet = []
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
                if name.startswith("resnet_encoder"):
                    resnet.append(param)
                elif self.pooling_weight_decay is not None and (
                    name.startswith("pooling")
                ):
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
        if resnet:
            resnet_params = {"params": resnet}
            if self.resnet_lr is not None:
                resnet_params["lr"] = self.resnet_lr
            if self.resnet_weight_decay is not None:
                resnet_params["weight_decay"] = self.resnet_weight_decay

            trainable_params.append(resnet_params)

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

    @property
    def sample_frequency(self) -> float:
        """Return the sampling frequency assumed by ``acoustic_feats``.

        Returns:
            Sampling frequency in hertz.
        """
        return self.acoustic_feats.sample_frequency

    def backbone_output_feats(self) -> int:
        """Return flattened ResNet channels and frequency dimension.

        Returns:
            Backbone feature dimension per frame.
        """
        return self._backbone_output_feats

    def freeze_backbone_feat_extractor(self) -> None:
        """Freeze the acoustic feature extractor."""
        self.acoustic_feats.freeze()

    def set_backbone_feat_extractor_in_eval_mode(self) -> None:
        """Put acoustic features into evaluation mode."""
        self.acoustic_feats.eval()

    def set_backbone_feat_extractor_in_train_mode(self) -> None:
        """Put acoustic features into training mode."""
        self.acoustic_feats.train()

    def _infer_backbone_layers_indices(self) -> None:
        """Determine which backbone layers to capture for feature inspection."""
        self.backbone_layers = [1, 2, 3, 4]

    def init_from_xvector(self, xvector_model: HyperTorchModel) -> None:
        """Initialize x-vector model backbone parameters from a pre-trained x-vector model.

        Args:
            xvector_model: Pre-trained x-vector model to use for initialization.
        """
        if not isinstance(xvector_model, RXVec):
            raise TypeError("xvector_model must be a Wav2ResNetXVector instance")
        feats = xvector_model.feats
        feats.spec_augment = self.acoustic_feats.spec_augment
        self.acoustic_feats = feats
        self.resnet_encoder.load_state_dict(
            xvector_model.xvector.encoder_net.state_dict()
        )

    def freeze_backbone(self) -> None:
        """Freeze the ResNet backbone."""
        self.resnet_encoder.freeze()

    def freeze_adapters(self) -> None:
        """Do nothing because this model does not define adapter modules."""

    def set_backbone_in_train_mode(self) -> None:
        """Put the ResNet backbone into training mode."""
        self.resnet_encoder.train()

    def set_backbone_in_eval_mode(self) -> None:
        """Put the ResNet backbone into evaluation mode."""
        self.resnet_encoder.eval()

    def set_adapters_in_train_mode(self) -> None:
        """Do nothing because this model does not define adapter modules."""

    def set_adapters_in_eval_mode(self) -> None:
        """Do nothing because this model does not define adapter modules."""

    def change_config(
        self,
        encoder_dropout_rate: Optional[float] = None,
        resnet_lr: Optional[float] = None,
        resnet_weight_decay: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Change model configuration at runtime.

        Args:
            encoder_dropout_rate: Optional dropout rate to apply in the ResNet
                encoder during fine-tuning.
            resnet_lr: Optional learning-rate override for backbone
                ``resnet_encoder`` parameters.
            resnet_weight_decay: Optional weight-decay override for backbone
                ``resnet_encoder`` parameters.
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

        super().change_config(**kwargs)

    def forward_backbone(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_hidden_feats: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        """Run acoustic front-end and resnet backbone.

        Args:
            x: Input waveform tensor shaped ``(batch, samples)``.
            x_lengths: Optional lengths per input example.
            return_hidden_feats: Whether to also return hidden layer activations.

        Returns:
            Tuple with backbone outputs, their lengths, hidden features, and hidden
            feature lengths (entries are ``None`` when unavailable).
        """
        with self._acoustic_feats_context:
            x, x_lengths = self.acoustic_feats(x, x_lengths)
            if not self.acoustic_feats.trans:
                x = x.transpose(1, 2)
            x = x.contiguous().view(x.size(0), 1, x.size(1), x.size(2))
            max_in_length = x.size(3)

        if return_hidden_feats:
            backbone_hidden_feats = self.resnet_encoder.forward_hid_feats(
                x,
                x_lengths,
                layers=self.backbone_layers,
                return_output=True,
            )
            backbone_hidden_feats, backbone_feats = backbone_hidden_feats
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

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable dictionary capturing constructor arguments.

        Args:
            no_class_name: Whether to omit the registered model class name.

        Returns:
            Dict[str, Any]: Configuration for acoustic features, backbone, and base
            ``XVectorP`` options.
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
        base_config = super().get_config(no_class_name=no_class_name)
        config = {
            "acoustic_feats": feats_cfg,
            "resnet_encoder": resnet_cfg,
            "resnet_lr": self.resnet_lr,
            "resnet_weight_decay": self.resnet_weight_decay,
        }
        config.update(base_config)
        return config

    @classmethod
    def load(
        cls,
        file_path: Optional[PathLike] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "ResNetXVectorP":
        """Instantiate a model from serialized configuration/state.

        Args:
            file_path: Optional path to a checkpoint bundle.
            cfg: Optional configuration dictionary to override disk contents.
            state_dict: Optional PyTorch state dictionary.

        Returns:
            ResNetXVectorP: Model with configuration/state restored.
        """
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Return only keyword args that match the constructor signature.

        Args:
            **kwargs: Candidate model configuration values.

        Returns:
            Keyword arguments accepted by ``ResNetXVectorP.__init__``.
        """
        return filter_func_args(ResNetXVectorP.__init__, kwargs)

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
            skip: Optional set of argument names to omit.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip = set(skip or ())
        if "acoustic_feats" not in skip:
            AudioFeatsMVN.add_class_args(parser, prefix="acoustic_feats")
        if "resnet_encoder" not in skip:
            RNF.add_class_args(parser, prefix="resnet_encoder")
        if "resnet_lr" not in skip:
            parser.add_argument(
                "--resnet-lr",
                type=float,
                default=None,
                help="optional learning-rate override for ResNet backbone parameters",
            )
        if "resnet_weight_decay" not in skip:
            parser.add_argument(
                "--resnet-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for ResNet backbone parameters",
            )
        XVectorP.add_class_args(parser, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Return fine-tuning keyword arguments accepted by ``change_config``.

        Args:
            **kwargs: Candidate fine-tuning configuration values.

        Returns:
            Keyword arguments accepted by ``ResNetXVectorP.change_config``.
        """
        base_args = XVectorP.filter_finetune_args(**kwargs)
        child_args = filter_func_args(ResNetXVectorP.change_config, kwargs)
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
            parser: ``ArgumentParser`` that receives the fine-tuning arguments.
            prefix: Optional namespace prefix for grouped argument registration.
            skip: Optional set of argument names to omit.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip = set(skip or ())
        if "encoder_dropout_rate" not in skip:
            parser.add_argument(
                "--encoder-dropout-rate",
                type=float,
                default=None,
                help="optional dropout rate for the ResNet encoder during fine-tuning",
            )
        if "resnet_lr" not in skip:
            parser.add_argument(
                "--resnet-lr",
                type=float,
                default=None,
                help="optional learning-rate override for ResNet backbone parameters",
            )
        if "resnet_weight_decay" not in skip:
            parser.add_argument(
                "--resnet-weight-decay",
                type=float,
                default=None,
                help="optional weight-decay override for ResNet backbone parameters",
            )
        XVectorP.add_finetune_args(parser, skip=skip)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

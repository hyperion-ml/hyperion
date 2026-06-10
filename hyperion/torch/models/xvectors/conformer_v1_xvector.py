"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...narchs import ConformerEncoderV1 as Encoder
from .xvector import XVector


class ConformerV1XVector(XVector):
    def __init__(
        self,
        encoder: Union[Dict[str, Any], Encoder],
        num_classes: int,
        pool_net: str = "mean+stddev",
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Union[Dict[str, Any], Any] = {"name": "relu", "inplace": True},
        loss_type: str = "arc-softmax",
        cos_scale: int = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        dropout_rate: float = 0,
        norm_layer: Optional[Any] = None,
        head_norm_layer: Optional[Any] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        head_use_norm: bool = True,
        head_use_in_norm: bool = False,
        head_hid_dim: int = 2048,
        head_bottleneck_dim: int = 256,
        proj_head_use_norm: bool = True,
        proj_head_norm_before: bool = True,
        embed_layer: int = 0,
        proj_feats: Optional[int] = None,
        head_type: str = "x-vector",
        bias_weight_decay: Optional[float] = None,
    ) -> None:
        if isinstance(encoder, dict):
            logging.info(f"making conformer encoder network={encoder}")
            encoder["in_time_dim"] = 2
            encoder["out_time_dim"] = 2
            encoder = Encoder(**encoder)
        else:
            encoder.in_time_dim = 2
            encoder.out_time_dim = 2

        super().__init__(
            encoder,
            num_classes,
            pool_net=pool_net,
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
            norm_layer=norm_layer,
            head_norm_layer=head_norm_layer,
            use_norm=use_norm,
            norm_before=norm_before,
            head_use_norm=head_use_norm,
            head_use_in_norm=head_use_in_norm,
            head_hid_dim=head_hid_dim,
            head_bottleneck_dim=head_bottleneck_dim,
            proj_head_use_norm=proj_head_use_norm,
            proj_head_norm_before=proj_head_norm_before,
            dropout_rate=dropout_rate,
            embed_layer=embed_layer,
            proj_feats=proj_feats,
            head_type=head_type,
            bias_weight_decay=bias_weight_decay,
        )

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        del base_config["encoder_cfg"]
        del base_config["in_feats"]

        encoder_cfg = self.encoder_net.get_config()
        del encoder_cfg["class_name"]
        config = {
            "encoder": encoder_cfg,
        }

        config.update(base_config)
        return config

    def change_config(
        self,
        encoder: Dict[str, Any],
        override_output: bool = False,
        override_dropouts: bool = False,
        dropout_rate: float = 0,
        num_classes: Optional[int] = None,
        loss_type: str = "arc-softmax",
        cos_scale: int = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 10,
        intertop_k: int = 5,
        intertop_margin: float = 0,
        num_subcenters: int = 2,
    ) -> None:
        super().change_config(
            override_output,
            False,
            dropout_rate,
            num_classes,
            loss_type,
            cos_scale,
            margin,
            margin_warmup_epochs,
            intertop_k,
            intertop_margin,
            num_subcenters,
        )
        if override_dropouts:
            logging.info("chaning x-vector head dropouts")
            self.classif_net.change_dropouts(dropout_rate)

        self.encoder_net.change_config(**encoder)

    @classmethod
    def load(
        cls,
        file_path: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> "ConformerV1XVector":
        cfg, state_dict = cls._load_cfg_state_dict(file_path, cfg, state_dict)
        try:
            del cfg["in_feats"]
        except:
            pass

        model = cls(**cfg)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        return model

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        base_args = XVector.filter_args(**kwargs)
        child_args = Encoder.filter_args(**kwargs["encoder"])

        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_class_args(parser, skip=set(["in_feats"]))
        Encoder.add_class_args(parser, prefix="encoder", skip=set())
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        base_args = XVector.filter_finetune_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["encoder"])
        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_finetune_args(parser)
        Encoder.add_finetune_args(parser, prefix="encoder", skip=set())

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_dino_teacher_args(**kwargs: Any) -> Dict[str, Any]:
        base_args = XVector.filter_dino_teacher_args(**kwargs)
        child_args = Encoder.filter_finetune_args(**kwargs["encoder"])
        base_args["encoder"] = child_args
        return base_args

    @staticmethod
    def add_dino_teacher_args(parser: Any, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        XVector.add_dino_teacher_args(parser)
        Encoder.add_finetune_args(parser, prefix="encoder", skip=set())

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

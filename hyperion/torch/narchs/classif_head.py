"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.nn import Linear

from ...utils.misc import filter_func_args
from ..layer_blocks import FCBlock
from ..layers import ActivationFactory as AF
from ..layers import ArcLossOutput, CosLossOutput
from ..layers import NormLayer1dFactory as NLF
from ..layers import SubCenterArcLossOutput
from .net_arch import NetArch


class ClassifHead(NetArch):
    """Classification head for x-vector style networks.

    Attributes:
       in_feats: Input feature dimension.
       num_classes: Number of output classes.
       embed_dim: Dimension of the embedding layer.
       num_embed_layers: Number of hidden layers.
       hid_act: Hidden activation configuration.
       loss_type: Loss function used by the head. Supported values are
          ``softmax``, ``cos-softmax``, ``arc-softmax``, and
          ``subcenter-arc-softmax``.
       cos_scale: Scale parameter for cosine-based losses.
       margin: Margin parameter for cosine-based losses.
       margin_warmup_epochs: Number of epochs used to anneal the margin from 0
          to the target value.
       intertop_k: Number of negative scores used by the InterTopK penalty.
       intertop_margin: InterTopK penalty value.
       num_subcenters: Number of sub-centers in subcenter losses.
       norm_layer: Normalization layer type. If ``None``, BatchNorm1d is used.
       use_norm: Whether to apply normalization layers.
       norm_before: If ``True``, normalization is applied before activation.
       use_in_norm: If ``True``, apply normalization at the input.
    """

    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        embed_dim: int = 256,
        num_embed_layers: int = 1,
        hid_act: Any = {"name": "relu", "inplace": True},
        loss_type: str = "arc-softmax",
        cos_scale: float = 64,
        margin: float = 0.3,
        margin_warmup_epochs: int = 0,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
        norm_layer: Optional[str] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        dropout_rate: float = 0,
        use_in_norm: bool = False,
    ) -> None:
        """Build the classification head.

        Args:
            in_feats: Input feature dimension.
            num_classes: Number of output classes.
            embed_dim: Embedding dimension of the hidden stack.
            num_embed_layers: Number of embedding blocks.
            hid_act: Hidden-layer activation configuration.
            loss_type: Classification loss variant.
            cos_scale: Scale factor for margin-based losses.
            margin: Margin for margin-based losses.
            margin_warmup_epochs: Warmup duration for the margin scheduler.
            intertop_k: Number of negatives used by the InterTopK penalty.
            intertop_margin: InterTopK margin value.
            num_subcenters: Number of sub-centers for subcenter losses.
            norm_layer: Normalization layer type.
            use_norm: Whether to enable normalization layers.
            norm_before: Whether normalization happens before activation.
            dropout_rate: Dropout rate used in the embedding stack.
            use_in_norm: Whether to apply normalization at the input.
        """
        super().__init__()
        assert num_embed_layers >= 1, "num_embed_layers (%d < 1)" % num_embed_layers

        self.num_embed_layers = num_embed_layers
        self.in_feats = in_feats
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.norm_layer = norm_layer
        self.use_in_norm = use_in_norm

        if use_norm:
            norm_groups = None
            if norm_layer == "group-norm":
                norm_groups = min(embed_dim // 8, 32)
            self._norm_layer = NLF.create(norm_layer, norm_groups)
        else:
            self._norm_layer = None

        self.use_norm = use_norm
        self.norm_before = norm_before

        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_k = intertop_k
        self.intertop_margin = intertop_margin
        self.num_subcenters = num_subcenters

        prev_feats = in_feats
        if self.use_in_norm:
            assert not self.norm_before
            self.in_norm = self._norm_layer(prev_feats)

        fc_blocks = []
        for i in range(num_embed_layers - 1):
            fc_blocks.append(
                FCBlock(
                    prev_feats,
                    embed_dim,
                    activation=hid_act,
                    dropout_rate=dropout_rate,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )
            prev_feats = embed_dim

        if loss_type != "softmax":
            act = None
        else:
            act = hid_act

        if self.use_in_norm:
            fc_blocks.append(
                FCBlock(prev_feats, embed_dim, activation=act, use_norm=False)
            )
        else:
            fc_blocks.append(
                FCBlock(
                    prev_feats,
                    embed_dim,
                    activation=act,
                    norm_layer=self._norm_layer,
                    use_norm=use_norm,
                    norm_before=norm_before,
                )
            )

        self.fc_blocks = nn.ModuleList(fc_blocks)

        # output layer
        if loss_type == "softmax":
            self.output = Linear(embed_dim, num_classes)
        elif loss_type == "cos-softmax":
            self.output = CosLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "arc-softmax":
            self.output = ArcLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "subcenter-arc-softmax":
            self.output = SubCenterArcLossOutput(
                embed_dim,
                num_classes,
                num_subcenters,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def rebuild_output_layer(
        self,
        num_classes: int,
        loss_type: str,
        cos_scale: float,
        margin: float,
        margin_warmup_epochs: int,
        intertop_k: int = 5,
        intertop_margin: float = 0.0,
        num_subcenters: int = 2,
    ) -> None:
        """Rebuild the output layer with a new classification-loss setup.

        Args:
            num_classes: Number of output classes.
            loss_type: Output loss variant to instantiate.
            cos_scale: Scale factor for margin-based losses.
            margin: Margin value for margin-based losses.
            margin_warmup_epochs: Number of epochs used to warm up the margin.
            intertop_k: Number of negatives used by the InterTopK penalty.
            intertop_margin: InterTopK margin value.
            num_subcenters: Number of sub-centers for subcenter losses.
        """
        embed_dim = self.embed_dim
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.cos_scale = cos_scale
        self.margin = margin
        self.margin_warmup_epochs = margin_warmup_epochs
        self.intertop_margin = intertop_margin
        self.num_subcenters = num_subcenters
        self.num_subcenters = num_subcenters

        if loss_type == "softmax":
            self.output = Linear(embed_dim, num_classes)
        elif loss_type == "cos-softmax":
            self.output = CosLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "arc-softmax":
            self.output = ArcLossOutput(
                embed_dim,
                num_classes,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        elif loss_type == "subcenter-arc-softmax":
            self.output = SubCenterArcLossOutput(
                embed_dim,
                num_classes,
                num_subcenters,
                cos_scale=cos_scale,
                margin=margin,
                margin_warmup_epochs=margin_warmup_epochs,
                intertop_k=intertop_k,
                intertop_margin=intertop_margin,
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def set_margin(self, margin: float) -> None:
        """Update the margin on the current output layer, if applicable.

        Args:
            margin: New margin value.
        """
        if self.loss_type == "softmax":
            return

        self.margin = margin
        self.output.margin = margin

    def set_margin_warmup_epochs(self, margin_warmup_epochs: int) -> None:
        """Update the margin warmup duration on the current output layer.

        Args:
            margin_warmup_epochs: New warmup duration.
        """
        if self.loss_type == "softmax":
            return

        self.margin_warmup_epochs = margin_warmup_epochs
        self.output.margin_warmup_epochs = margin_warmup_epochs

    def set_cos_scale(self, cos_scale: float) -> None:
        """Update the scale factor used by the current output layer.

        Args:
            cos_scale: New scale factor.
        """
        if self.loss_type == "softmax":
            return

        self.cos_scale = cos_scale
        self.output.cos_scale = cos_scale

    def set_intertop_k(self, intertop_k: int) -> None:
        """Update the InterTopK selection size on the current output layer.

        Args:
            intertop_k: New InterTopK count.
        """
        if self.loss_type == "softmax":
            return

        self.intertop_k = intertop_k
        self.output.intertop_k = intertop_k

    def set_intertop_margin(self, intertop_margin: float) -> None:
        """Update the InterTopK margin on the current output layer.

        Args:
            intertop_margin: New InterTopK margin value.
        """
        if self.loss_type == "softmax":
            return

        self.intertop_margin = intertop_margin
        self.output.intertop_margin = intertop_margin

    def set_num_subcenters(self, num_subcenters: int) -> None:
        """Update the subcenter count when using subcenter arc-softmax.

        Args:
            num_subcenters: New number of sub-centers.
        """
        if not self.loss_type == "subcenter-arc-softmax":
            return

        self.num_subcenters = num_subcenters
        self.output.num_subcenters = num_subcenters

    def update_margin(self, epoch: int) -> None:
        """Propagate the current epoch or step to the output margin scheduler.

        Args:
            epoch: Current epoch or step index.
        """
        if hasattr(self.output, "update_margin"):
            self.output.update_margin(epoch)

    def freeze_layers(self, layer_list: Sequence[int]) -> None:
        """Disable gradient updates for the selected embedding layers.

        Args:
            layer_list: Layer indices to freeze.
        """
        for l in layer_list:
            for param in self.fc_blocks[l].parameters():
                param.requires_grad = False

    def put_layers_in_eval_mode(self, layer_list: Sequence[int]) -> None:
        """Put the selected embedding layers into evaluation mode.

        Args:
            layer_list: Layer indices to switch to evaluation mode.
        """
        for l in layer_list:
            self.fc_blocks[l].eval()

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute classification logits for the input batch.

        Args:
            x: Input tensor of shape ``(batch, in_feats)``.
            y: Optional target labels for margin-based losses.

        Returns:
            Logit tensor with shape ``(batch, num_classes)``.
        """
        if not torch.all(torch.isfinite(x)):
            logging.warning("non-finite x-in=%f", torch.mean(x))
        if self.use_in_norm:
            x = self.in_norm(x)

        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)

        if self.loss_type == "softmax":
            y = self.output(x)
        else:
            y = self.output(x, y)

        if not torch.all(torch.isfinite(y)):
            logging.warning("non-finite y-=%f", torch.mean(y))
        return y

    def forward_hid_feats(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_layers: Optional[Sequence[int]] = None,
        return_logits: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Return selected hidden activations and, optionally, the logits.

        Args:
            x: Input tensor of shape ``(batch, in_feats)``.
            y: Optional target labels for margin-based losses.
            return_layers: Layer indices whose activations should be returned.
            return_logits: Whether to also return the final logits.

        Returns:
            A pair ``(hidden_activations, logits)``. The second element is
            ``None`` when ``return_logits`` is ``False``.
        """
        assert return_layers is not None or return_logits
        if return_layers is None:
            return_layers = []

        if self.use_in_norm:
            x = self.in_norm(x)

        h = []
        for l in range(self.num_embed_layers):
            x = self.fc_blocks[l](x)
            if l in return_layers:
                h.append(x)

        if self.loss_type == "softmax":
            y = self.output(x)
        else:
            y = self.output(x, y)

        if return_logits:
            return h, y
        return h, None

    def extract_embed(self, x: torch.Tensor, embed_layer: int = 0) -> torch.Tensor:
        """Extract the embedding produced by a specific block index.

        Args:
            x: Input tensor of shape ``(batch, in_feats)``.
            embed_layer: Index of the embedding layer to inspect.

        Returns:
            Embedding tensor from the requested layer.
        """
        if self.use_in_norm:
            x = self.in_norm(x)

        for l in range(embed_layer):
            x = self.fc_blocks[l](x)

        if self.loss_type == "softmax" or embed_layer < self.num_embed_layers:
            y = self.fc_blocks[embed_layer].forward_linear(x)
        else:
            y = self.fc_blocks[l](x)
        return y

    def compute_prototype_affinity(self) -> torch.Tensor:
        """Compute cosine affinity between the current class prototypes.

        Returns:
            Cosine affinity matrix of shape ``(num_classes, num_classes)``.
        """
        if self.loss_type != "softmax":
            return self.output.compute_prototype_affinity()

        kernel = self.output.weight  # (num_classes, feat_dim)
        kernel = kernel / torch.linalg.norm(kernel, 2, dim=1, keepdim=True)
        return torch.mm(kernel, kernel.transpose(0, 1))

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration for the head.

        Args:
            no_class_name: If ``True``, omit the class name from the config.

        Returns:
            Dictionary containing the constructor arguments.
        """
        hid_act = AF.get_config(self.fc_blocks[0].activation)

        config = {
            "in_feats": self.in_feats,
            "num_classes": self.num_classes,
            "embed_dim": self.embed_dim,
            "num_embed_layers": self.num_embed_layers,
            "hid_act": hid_act,
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
            "dropout_rate": self.dropout_rate,
            "use_in_norm": self.use_in_norm,
        }

        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter constructor keyword arguments to the supported subset.

        Args:
            **kwargs: Candidate constructor keyword arguments.

        Returns:
            Dictionary with unsupported keys removed and aliases normalized.
        """
        if "wo_norm" in kwargs:
            kwargs["use_norm"] = not kwargs["wo_norm"]
            del kwargs["wo_norm"]

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        args = filter_func_args(ClassifHead.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register CLI arguments for configuring the classification head.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used to nest the arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

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
            default=0,
            type=int,
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
            "--use-in-norm",
            default=False,
            action=ActionYesNo,
            help="batch normalizaton in the classif head input",
        )

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

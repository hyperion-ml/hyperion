"""
 Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from ...layers import ActivationFactory as AF
from ...narchs import FCNetV2
from .splda import SPLDA


class FCNSPLDA(SPLDA):
    """SPLDA backend with an FCNetV2 preprocessor.

    Attributes:
        preprocessor: Fully connected feature extractor applied before PLDA.
        in_feats: Input feature dimension of the preprocessor.
        num_layers: Number of hidden fully connected layers.
        hid_feats: Hidden feature width specification.
        hid_act: Hidden activation configuration.
        dropout_rate: Dropout probability used by the preprocessor.
        norm_layer: Normalization-layer specification used by the preprocessor.
        use_norm: Whether normalization layers are enabled in the preprocessor.
    """

    def __init__(
        self,
        in_feats: int,
        num_layers: int,
        hid_feats: Any,
        x_dim: int,
        y_dim: int,
        mu: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        num_classes: int = 0,
        x_ref: Optional[torch.Tensor] = None,
        p_tar: float = 0.05,
        margin_multi: float = 0.3,
        margin_tar: float = 0.3,
        margin_non: float = 0.3,
        margin_warmup_epochs: int = 10,
        adapt_margin: bool = False,
        adapt_gamma: float = 0.99,
        lnorm: bool = False,
        hid_act: Any = {"name": "relu6", "inplace": True},
        dropout_rate: float = 0,
        norm_layer: Any = "batch-norm",
        use_norm: bool = True,
        var_floor: float = 1e-5,
        prec_floor: float = 1e-5,
    ) -> None:
        """Initialize the FCNSPLDA model.

        Args:
            in_feats: Input feature dimension of the preprocessor.
            num_layers: Number of hidden fully connected layers.
            hid_feats: Hidden feature specification for the preprocessor.
            x_dim: Output embedding dimension of the preprocessor.
            y_dim: Speaker-factor dimension for SPLDA.
            mu: Optional global mean vector.
            V: Optional loading matrix.
            W: Optional within-class precision matrix.
            num_classes: Number of reference classes for multi-class scoring.
            x_ref: Optional reference embeddings used for multi-class scoring.
            p_tar: Target prior probability.
            margin_multi: Multi-class margin value.
            margin_tar: Binary target margin value.
            margin_non: Binary non-target margin value.
            margin_warmup_epochs: Number of epochs used to warm up margins.
            adapt_margin: Whether to adapt margins from observed scores.
            adapt_gamma: Exponential moving-average factor for adaptation.
            lnorm: Whether to length-normalize embeddings before scoring.
            hid_act: Hidden activation specification for the preprocessor.
            dropout_rate: Dropout probability used by the preprocessor.
            norm_layer: Normalization-layer specification for the preprocessor.
            use_norm: Whether normalization is enabled in the preprocessor.
            var_floor: Lower bound for variance-like quantities.
            prec_floor: Lower bound for precision-like quantities.
        """
        # dd = locals()
        # del dd["self"]
        # print(dd, flush=True)
        # in_feats = 139
        # num_layers = 3
        # hid_feats = 128
        # hid_act = "relu6"
        # dropout_rate = 0
        preprocessor = FCNetV2(
            num_layers,
            in_feats,
            hid_feats,
            x_dim,
            hid_act=hid_act,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            use_norm=use_norm,
            norm_before=True,
        )

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            mu=mu,
            V=V,
            W=W,
            num_classes=num_classes,
            x_ref=x_ref,
            p_tar=p_tar,
            margin_multi=margin_multi,
            margin_tar=margin_tar,
            margin_non=margin_non,
            margin_warmup_epochs=margin_warmup_epochs,
            adapt_margin=adapt_margin,
            adapt_gamma=adapt_gamma,
            lnorm=lnorm,
            var_floor=var_floor,
            prec_floor=prec_floor,
            preprocessor=preprocessor,
        )

    @property
    def in_feats(self) -> int:
        """Return the input feature dimension of the preprocessor."""
        return self.preprocessor.in_units

    @property
    def num_layers(self) -> int:
        """Return the number of hidden layers in the preprocessor."""
        return self.preprocessor.num_blocks

    @property
    def hid_feats(self) -> Any:
        """Return the hidden-layer configuration of the preprocessor."""
        return self.preprocessor.hid_units

    @property
    def hid_act(self) -> Any:
        """Return the hidden activation configuration."""
        hid_act = AF.get_config(self.preprocessor.blocks[0].activation)
        return hid_act

    @property
    def dropout_rate(self) -> float:
        """Return the preprocessor dropout rate."""
        return self.preprocessor.dropout_rate

    @property
    def norm_layer(self) -> Any:
        """Return the preprocessor normalization-layer configuration."""
        return self.preprocessor.norm_layer

    @property
    def use_norm(self) -> bool:
        """Return whether the preprocessor uses normalization."""
        return self.preprocessor.use_norm

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Returns:
            Configuration dictionary that can be used to reconstruct the model.
        """
        config = {
            "in_feats": self.in_feats,
            "num_layers": self.num_layers,
            "hid_feats": self.hid_feats,
            "hid_act": self.hid_act,
            "dropout_rate": self.dropout_rate,
            "norm_layer": self.norm_layer,
            "use_norm": self.use_norm,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

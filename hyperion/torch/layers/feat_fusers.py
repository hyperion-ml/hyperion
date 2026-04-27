"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class FeatFuser(nn.Module):
    """Base class for feature fusion modules.

    Attributes:
      None.
    """

    def __init__(self) -> None:
        """Initializes the feature fuser base module.

        Args:
          None.

        Returns:
          None.
        """
        super().__init__()


class _ProjFeatFuser(FeatFuser):
    """Base feature fuser with optional output projection.

    Attributes:
      feat_dim: Input feature dimension expected by the optional projection.
      proj_dim: Output feature dimension of the optional projection.
      feat_proj: Optional linear projection applied after fusion.
    """

    def __init__(
        self,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> None:
        """Initializes the optional projection used by derived fusers.

        Args:
          feat_dim: Input feature dimension for the projection layer.
          proj_dim: Output feature dimension for the projection layer.
          proj_bias: Whether the projection layer uses a bias term.

        Returns:
          None.
        """
        super().__init__()
        if (feat_dim is None) != (proj_dim is None):
            raise ValueError(
                "feat_dim and proj_dim must both be set or both be None "
                f"(got feat_dim={feat_dim}, proj_dim={proj_dim})"
            )
        if feat_dim is not None and feat_dim <= 0:
            raise ValueError(f"feat_dim must be > 0, got {feat_dim}")
        if proj_dim is not None and proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {proj_dim}")

        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.feat_proj: Optional[nn.Linear] = None
        if feat_dim is not None and proj_dim is not None:
            self.feat_proj = nn.Linear(feat_dim, proj_dim, bias=proj_bias)


class LastFeatFuser(_ProjFeatFuser):
    """Selects the last feature tensor from a sequence of features.

    Attributes:
      feat_dim: Input feature dimension expected by the optional projection.
      proj_dim: Output feature dimension of the optional projection.
      feat_proj: Optional linear projection applied after selecting the last feature.
    """

    def __init__(
        self,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> None:
        """Initializes the last-feature fuser.

        Args:
          feat_dim: Input feature dimension for the projection layer.
          proj_dim: Output feature dimension for the projection layer.
          proj_bias: Whether the projection layer uses a bias term.

        Returns:
          None.
        """
        super().__init__(feat_dim, proj_dim, proj_bias)

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        """Selects and optionally projects the last feature tensor.

        Args:
          feats: Sequence of feature tensors with compatible shape.

        Returns:
          Fused tensor, equal to the last element in ``feats`` with optional projection.
        """
        if len(feats) == 0:
            raise ValueError("feats must contain at least one tensor")
        feats = feats[-1]
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class WeightedAvgFeatFuser(_ProjFeatFuser):
    """Fuses features with trainable softmax-normalized scalar weights.

    Attributes:
      num_feats: Number of feature tensors expected in the input sequence.
      feat_fuser: Learnable logits used to compute softmax fusion weights.
      feat_dim: Input feature dimension expected by the optional projection.
      proj_dim: Output feature dimension of the optional projection.
      feat_proj: Optional linear projection applied after weighted averaging.
    """

    def __init__(
        self,
        num_feats: int,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> None:
        """Initializes the weighted-average feature fuser.

        Args:
          num_feats: Number of feature tensors to fuse.
          feat_dim: Input feature dimension for the projection layer.
          proj_dim: Output feature dimension for the projection layer.
          proj_bias: Whether the projection layer uses a bias term.

        Returns:
          None.
        """
        if num_feats is None:
            raise ValueError("num_feats must not be None")
        if num_feats <= 0:
            raise ValueError(f"num_feats must be > 0, got {num_feats}")
        super().__init__(feat_dim, proj_dim, proj_bias)
        self.num_feats = num_feats
        self.feat_fuser = nn.Parameter(torch.zeros(num_feats))

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        """Computes a weighted average across the feature sequence.

        Args:
          feats: Sequence of feature tensors with matching shape.

        Returns:
          Fused tensor obtained from the weighted average with optional projection.
        """
        if len(feats) != self.num_feats:
            raise ValueError(
                f"expected {self.num_feats} feature tensors, got {len(feats)}"
            )
        feats = torch.stack(feats, dim=-1)
        norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
        feats = torch.sum(feats * norm_weights, dim=-1)
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class LinearFeatFuser(_ProjFeatFuser):
    """Fuses features using a learnable linear combination.

    Attributes:
      num_feats: Number of feature tensors expected in the input sequence.
      feat_fuser: Linear layer that combines feature tensors along the feature-list axis.
      feat_dim: Input feature dimension expected by the optional projection.
      proj_dim: Output feature dimension of the optional projection.
      feat_proj: Optional linear projection applied after linear fusion.
    """

    def __init__(
        self,
        num_feats: int,
        feat_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> None:
        """Initializes the linear feature fuser.

        Args:
          num_feats: Number of feature tensors to fuse.
          feat_dim: Input feature dimension for the projection layer.
          proj_dim: Output feature dimension for the projection layer.
          proj_bias: Whether the projection layer uses a bias term.

        Returns:
          None.
        """
        if num_feats is None:
            raise ValueError("num_feats must not be None")
        if num_feats <= 0:
            raise ValueError(f"num_feats must be > 0, got {num_feats}")
        super().__init__(feat_dim, proj_dim, proj_bias)
        self.num_feats = num_feats
        self.feat_fuser = nn.Linear(num_feats, 1, bias=False)
        self.feat_fuser.weight.data = torch.ones(1, num_feats) / num_feats

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        """Applies a linear fusion across the feature sequence.

        Args:
          feats: Sequence of feature tensors with matching shape.

        Returns:
          Fused tensor after linear combination with optional projection.
        """
        if len(feats) != self.num_feats:
            raise ValueError(
                f"expected {self.num_feats} feature tensors, got {len(feats)}"
            )
        feats = torch.stack(feats, dim=-1)
        feats = self.feat_fuser(feats).squeeze(dim=-1)
        if self.feat_proj is not None:
            feats = self.feat_proj(feats)

        return feats


class CatFeatFuser(FeatFuser):
    """Concatenates features and projects them with a linear layer.

    Attributes:
      num_feats: Number of feature tensors expected in the input sequence.
      feat_dim: Dimension of each input feature tensor.
      proj_dim: Output dimension after concatenation and projection.
      proj_bias: Whether the projection layer uses a bias term.
      feat_fuser: Linear projection applied to concatenated features.
    """

    def __init__(
        self,
        num_feats: int,
        feat_dim: int,
        proj_dim: Optional[int] = None,
        proj_bias: bool = True,
    ) -> None:
        """Initializes the concatenation-based feature fuser.

        Args:
          num_feats: Number of feature tensors to concatenate.
          feat_dim: Dimension of each input feature tensor.
          proj_dim: Output projection dimension. If ``None``, uses ``feat_dim``.
          proj_bias: Whether the projection layer uses a bias term.

        Returns:
          None.
        """
        super().__init__()
        if num_feats is None:
            raise ValueError("num_feats must not be None")
        if feat_dim is None:
            raise ValueError("feat_dim must not be None")
        if num_feats <= 0:
            raise ValueError(f"num_feats must be > 0, got {num_feats}")
        if feat_dim <= 0:
            raise ValueError(f"feat_dim must be > 0, got {feat_dim}")
        if proj_dim is not None and proj_dim <= 0:
            raise ValueError(f"proj_dim must be > 0, got {proj_dim}")
        self.num_feats = num_feats
        self.feat_dim = feat_dim
        if proj_dim is None:
            proj_dim = feat_dim
        self.proj_dim = proj_dim
        self.proj_bias = proj_bias
        self.feat_fuser = nn.Linear(num_feats * feat_dim, proj_dim, bias=proj_bias)

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        """Concatenates the input features and applies the projection layer.

        Args:
          feats: Sequence of feature tensors to concatenate on the last dimension.

        Returns:
          Fused tensor after concatenation and projection.
        """
        if len(feats) != self.num_feats:
            raise ValueError(
                f"expected {self.num_feats} feature tensors, got {len(feats)}"
            )
        feats = torch.cat(feats, dim=-1)
        feats = self.feat_fuser(feats)
        return feats

"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Tuple

import torch
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..layers import AudioFeatsFactory as AFF
from ..layers import MeanVarianceNorm as MVN
from ..layers import SpecAugment
from .net_arch import NetArch


class AudioFeatsMVN(NetArch):
    """Audio feature frontend with optional short-term MVN and SpecAugment.

    The module applies:
    1. Feature extraction from raw waveform (`audio_feats`).
    2. Optional SpecAugment (before or after MVN).
    3. Optional short-term mean/variance normalization (`mvn`).
    4. Optional output transpose to channel-first layout.

    Attributes:
        audio_feats_cfg: Filtered constructor config used to build `audio_feats`.
        audio_feats: Acoustic feature extractor created by `AudioFeatsFactory`.
        mvn_cfg: Optional filtered constructor config used to build `mvn`.
        mvn: Optional short-term mean/variance normalization layer.
        spec_augment_cfg: Optional filtered constructor config used to build `spec_augment`.
        spec_augment: Optional SpecAugment module.
        trans: If `True`, output layout is `(B, C, T)`; otherwise `(B, T, C)`.
        aug_after_mvn: If `True`, applies SpecAugment after MVN; otherwise before MVN.
    """

    def __init__(
        self,
        audio_feats: Dict[str, Any],
        mvn: Optional[Dict[str, Any]] = None,
        spec_augment: Optional[Dict[str, Any]] = None,
        trans: bool = False,
        aug_after_mvn: bool = False,
    ) -> None:
        """Build the audio frontend.

        Args:
            audio_feats: Configuration dictionary for `AudioFeatsFactory`.
            mvn: Optional configuration dictionary for `MeanVarianceNorm`.
            spec_augment: Optional configuration dictionary for `SpecAugment`.
            trans: If `True`, returns features as `(B, C, T)` instead of `(B, T, C)`.
            aug_after_mvn: If `True`, applies SpecAugment after MVN; otherwise before MVN.
        """
        super().__init__()

        audio_feats = AFF.filter_args(**audio_feats)
        self.audio_feats_cfg = audio_feats
        self.audio_feats = AFF.create(**audio_feats)

        self.mvn = None
        self.mvn_cfg = None
        if mvn is not None:
            mvn = MVN.filter_args(**mvn)
            self.mvn_cfg = mvn
            if (
                ("norm_mean" in mvn)
                and mvn["norm_mean"]
                or ("norm_var" in mvn)
                and mvn["norm_var"]
            ):
                self.mvn = MVN(**mvn)

        self.spec_augment = None
        self.spec_augment_cfg = None
        if spec_augment is not None:
            spec_augment = SpecAugment.filter_args(**spec_augment)
            self.spec_augment_cfg = spec_augment
            self.spec_augment = SpecAugment(**spec_augment)

        self.trans = trans
        self.aug_after_mvn = aug_after_mvn

    @property
    def out_feats(self) -> int:
        """Output feature dimension produced by the frontend."""
        return self.audio_feats.out_feats

    @property
    def sample_frequency(self) -> float:
        """Sampling rate expected by the feature extractor, in Hz."""
        return self.audio_feats.fs

    @property
    def fs(self) -> float:
        """Alias of `sample_frequency`."""
        return self.audio_feats.fs

    @property
    def frame_length(self) -> float:
        """Frame length used by the feature extractor."""
        return self.audio_feats.frame_length

    @property
    def frame_shift(self) -> float:
        """Frame shift (hop) used by the feature extractor."""
        return self.audio_feats.frame_shift

    @staticmethod
    def _compute_feat_lengths(
        x_lengths: Optional[torch.Tensor], max_samples: int, max_frames: int
    ) -> Optional[torch.Tensor]:
        """Map waveform lengths (samples) to feature lengths (frames)."""
        if x_lengths is None:
            return None

        return torch.div(x_lengths * max_frames, max_samples, rounding_mode="floor")

    def forward(
        self, x: torch.Tensor, x_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute frontend features and aligned feature lengths.

        Args:
            x: Input waveform batch with shape `(B, S)` where `S` is samples.
            x_lengths: Optional valid waveform lengths in samples, shape `(B,)`.

        Returns:
            Tuple `(f, f_lengths)` where:
            - `f` is the output feature tensor, shape `(B, T, C)` or `(B, C, T)` when `trans=True`.
            - `f_lengths` is the optional feature-length tensor in frames, shape `(B,)`.
        """
        f = self.audio_feats(x)
        f_lengths = self._compute_feat_lengths(x_lengths, x.size(-1), f.size(1))
        if self.spec_augment is not None and not self.aug_after_mvn:
            f = self.spec_augment(f, f_lengths)

        if self.mvn is not None:
            f = self.mvn(f, f_lengths)

        if self.spec_augment is not None and self.aug_after_mvn:
            f = self.spec_augment(f, f_lengths)

        if self.trans:
            f = f.transpose(1, 2).contiguous()

        return f, f_lengths

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return serialized constructor configuration."""
        config = {
            "audio_feats": self.audio_feats_cfg,
            "mvn": self.mvn_cfg,
            "spec_augment": self.spec_augment_cfg,
            "trans": self.trans,
            "aug_after_mvn": self.aug_after_mvn,
        }
        base_config = super().get_config(no_class_name=no_class_name)
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter kwargs to those accepted by this class constructor."""
        valid_args = ("audio_feats", "mvn", "spec_augment", "trans", "aug_after_mvn")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register CLI arguments for `AudioFeatsMVN`.

        Args:
            parser: Target argument parser.
            prefix: Optional top-level namespace to nest the arguments under.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AFF.add_class_args(parser, prefix="audio_feats")
        MVN.add_class_args(parser, prefix="mvn")
        SpecAugment.add_class_args(parser, prefix="spec_augment")
        parser.add_argument(
            "--aug-after-mvn",
            default=False,
            action=ActionYesNo,
            help=(
                "Apply SpecAugment after short-term mean/variance normalization "
                "(default: before MVN)."
            ),
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional, Union

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch
from jsonargparse import ActionParser, ArgumentParser

from ...hyper_torch_model import HyperTorchModel
from ...narchs import AudioFeatsMVN
from ...utils import remove_silence
from ..transducer import RNNTransducer, RNNTransducerOutput


class Wav2RNNTransducer(HyperTorchModel):
    """Base class for waveform-to-RNNT wrappers with acoustic frontends.

    Attributes:
      feats: Acoustic feature extractor instance or configuration dictionary.
      transducer: Backend RNN-T transducer model.
    """

    def __init__(
        self,
        feats: Union[Dict[str, Any], AudioFeatsMVN],
        transducer: RNNTransducer,
    ) -> None:
        """Initializes the waveform-to-transducer wrapper.

        Args:
          feats: Acoustic feature extractor instance or configuration dictionary.
          transducer: Backend RNN-T model that consumes acoustic features.
        """

        super().__init__()

        if isinstance(feats, dict):
            feats = AudioFeatsMVN.filter_args(**feats)
            feats["trans"] = False
            feats = AudioFeatsMVN(**feats)
        else:
            assert isinstance(feats, AudioFeatsMVN)

        self.feats = feats
        self.transducer = transducer

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: k2.RaggedTensor,
        vad_samples: Optional[torch.Tensor] = None,
        vad_feats: Optional[torch.Tensor] = None,
    ) -> RNNTransducerOutput:
        """Extracts features and computes RNNT losses.

        Args:
          x: Input waveform tensor.
          x_lengths: Number of valid samples in each waveform.
          y: Ragged tensor containing target token sequences.
          vad_samples: Optional voiced-sample mask used to trim waveform silence.
          vad_feats: Optional voiced-frame mask used to trim feature silence.

        Returns:
          RNN-T output container produced by the backend transducer.
        """

        if vad_samples is not None:
            x, x_lengths = remove_silence(x, vad_samples, x_lengths)
        feats, feat_lengths = self.feats(x, x_lengths)
        if vad_feats is not None:
            feats, feat_lengths = remove_silence(feats, vad_feats, feat_lengths)

        return self.transducer(feats, feat_lengths, y)

    def set_train_mode(self, mode: str) -> None:
        """Delegates train-mode selection to the backend transducer.

        Args:
          mode: Training mode understood by the backend transducer.
        """
        self.transducer.set_train_mode(mode)

    def get_config(self) -> Dict[str, Any]:
        """Serializes the wrapper configuration.

        Returns:
          Configuration dictionary suitable for reconstruction.
        """
        feat_cfg = self.feats.get_config()
        transducer_cfg = self.transducer.get_config()
        config = {
            "feats": feat_cfg,
            "transducer": transducer_cfg,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments from a configuration dictionary.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Subset of arguments accepted by this wrapper.
        """
        valid_args = (
            "feats",
            "transducer",
        )

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds wrapper CLI arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        AudioFeatsMVN.add_class_args(parser, prefix="feats")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

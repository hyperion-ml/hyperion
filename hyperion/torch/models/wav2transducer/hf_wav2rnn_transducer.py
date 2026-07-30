"""
Copyright 2022 Johns Hopkins University  (Author: Yen-Ju Lu)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import contextlib
import logging
from typing import Any, ContextManager, Dict, List, Optional, Set, Union

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch
import torch.nn as nn
from jsonargparse import ActionParser, ArgumentParser

from ...hyper_torch_model import HyperTorchModel
from ...utils import remove_silence
from ..transducer import RNNTransducer, RNNTransducerOutput


class HFWav2RNNTransducer(HyperTorchModel):
    """Base class for RNN-T models backed by Hugging Face features.

    Attributes:
       hf_feats: Hugging Face feature-extractor wrapper.
       transducer: Backend transducer model.
       feat_fusion_start: First hidden-state layer used for fusion.
       feat_fusion_method: Method used to fuse hidden-state layers.
    """

    def __init__(
        self,
        hf_feats: HyperTorchModel,
        transducer: Union[Dict[str, Any], HyperTorchModel],
        feat_fusion_start: int = 0,
        feat_fusion_method: str = "weighted-avg",
    ) -> None:
        """Initializes the wrapper.

        Args:
          hf_feats: Hugging Face feature-extractor wrapper.
          transducer: Backend transducer instance or configuration dictionary.
          feat_fusion_start: First hidden-state layer used for fusion.
          feat_fusion_method: Hidden-state fusion method.
        """

        super().__init__()
        self.hf_feats = hf_feats
        if isinstance(transducer, dict):
            transducer["rnnt_decoder"]["in_feats"] = hf_feats.hidden_size
            if "class_name" in transducer:
                del transducer["class_name"]

            transducer["encoder"] = None
            transducer = RNNTransducer(**transducer)
        else:
            assert isinstance(transducer, RNNTransducer)
            if transducer.encoder is None:
                assert transducer.rnnt_decoder.in_feats == hf_feats.hidden_size

        self.transducer = transducer
        self.feat_fusion_start = feat_fusion_start
        self.feat_fusion_method = feat_fusion_method
        self._hf_context = contextlib.nullcontext()
        self._make_fuser()

    def _make_fuser(self) -> None:
        """Creates the hidden-state fusion module for Hugging Face features."""
        if self.feat_fusion_method == "last":
            self.feat_fuser = None
            return

        num_layers = self.hf_feats.num_encoder_layers + 1 - self.feat_fusion_start
        layer_dim = self.hf_feats.hidden_size
        if self.feat_fusion_method == "weighted-avg":
            self.feat_fuser = nn.Parameter(torch.zeros(num_layers))
        elif self.feat_fusion_method == "linear":
            self.feat_fuser = nn.Linear(num_layers, 1, bias=False)
            self.feat_fuser.weight.data = torch.ones(1, num_layers) / num_layers
        elif self.feat_fusion_method == "cat":
            self.feat_fuser = nn.Linear(num_layers * layer_dim, layer_dim, bias=False)

    def _fuse_hid_feats(self, hid_feats: List[torch.Tensor]) -> torch.Tensor:
        """Fuses hidden features from the Hugging Face model.

        Args:
          hid_feats: Hidden-state tensors from the Hugging Face model.

        Returns:
          Fused feature tensor.
        """
        if len(hid_feats) == 1:
            # There is only one layer of features
            return hid_feats[0]

        hid_feats = hid_feats[self.feat_fusion_start :]
        if self.feat_fusion_method == "weighted-avg":
            hid_feats = torch.stack(hid_feats, dim=-1)
            norm_weights = nn.functional.softmax(self.feat_fuser, dim=-1)
            feats = torch.sum(hid_feats * norm_weights, dim=-1)
        elif self.feat_fusion_method == "linear":
            hid_feats = torch.stack(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats).squeeze(dim=-1)
        elif self.feat_fusion_method == "cat":
            hid_feats = torch.cat(hid_feats, dim=-1)
            feats = self.feat_fuser(hid_feats)
        elif self.feat_fusion_method == "last":
            feats = hid_feats[-1]

        return feats

    def forward_feats(
        self,
        x: torch.Tensor,
        x_lengths: Optional[torch.Tensor],
        return_feat_layers: Optional[List[int]] = None,
        chunk_length: int = 0,
        detach_chunks: bool = False,
    ) -> tuple[torch.Tensor, Optional[List[torch.Tensor]], torch.Tensor]:
        """Extracts and optionally fuses Hugging Face hidden features.

        Args:
          x: Input waveform tensor.
          x_lengths: Number of valid samples in each waveform.
          return_feat_layers: Optional hidden-state layer indices to return.
          chunk_length: Optional chunk length forwarded to the HF frontend.
          detach_chunks: Whether chunk outputs should be detached.

        Returns:
          Tuple with fused features, optional selected hidden layers, and
          feature lengths.
        """
        return_hid_states = (
            False
            if return_feat_layers is None and self.feat_fusion_method == "last"
            else True
        )
        with self._hf_context:
            hf_output = self.hf_feats(
                x,
                x_lengths,
                return_hid_states=return_hid_states,
                chunk_length=chunk_length,
                detach_chunks=detach_chunks,
            )
        feat_lengths = hf_output["hidden_states_lengths"]
        if return_hid_states:
            hid_feats = hf_output["hidden_states"]
            feats = self._fuse_hid_feats(hid_feats)
        else:
            hid_feats = None
            feats = hf_output["last_hidden_state"]

        feats = feats.transpose(1, 2)
        if return_feat_layers is not None:
            # add hidden feats from wav2vec to the output. We transpose to be (batch, C, time)
            # as the hidden features expected by the backend transducer.
            hid_feats = [
                f.transpose(1, 2)
                for i, f in enumerate(hid_feats)
                if i in return_feat_layers
            ]
        else:
            hid_feats = None

        return feats, hid_feats, feat_lengths

    def forward(
        self,
        x: torch.Tensor,
        y: k2.RaggedTensor,
        x_lengths: Optional[torch.Tensor] = None,
        return_feat_layers: Optional[List[int]] = None,
        return_logits: bool = True,
    ) -> RNNTransducerOutput:
        """Runs the Hugging Face frontend and backend transducer.

        Args:
          x: Input waveform tensor.
          y: Ragged tensor containing target token sequences.
          x_lengths: Number of valid samples in each waveform.
          return_feat_layers: Optional hidden-state layer indices to attach to
            the output.
          return_logits: Unused compatibility argument.

        Returns:
          RNN-T output container, optionally augmented with selected hidden
          features.
        """
        feats, hid_feats, feat_lengths = self.forward_feats(
            x, x_lengths, return_feat_layers
        )

        feats = feats.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)
        output = self.transducer(
            feats,
            feat_lengths,
            y,
        )

        if return_feat_layers:
            output.h_feats = hid_feats

        return output

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        decoding_method: str = "time_sync_beam_search",
        beam_width: int = 5,
        max_sym_per_frame: int = 3,
        max_sym_per_utt: int = 1000,
    ) -> List[List[int]]:
        """Decodes token sequences from waveform input.

        Args:
          x: Input waveform tensor.
          x_lengths: Number of valid samples in each waveform.
          decoding_method: Decoding algorithm to use.
          beam_width: Beam width for beam-search decoders.
          max_sym_per_frame: Maximum number of symbols the RNNT can emit in
            one frame.
          max_sym_per_utt: Maximum number of emitted symbols per utterance.

        Returns:
          One decoded token-id sequence per input utterance.
        """

        feats, _, feat_lengths = self.forward_feats(x, x_lengths)

        feats = feats.permute(0, 2, 1)  # (N, C, T) ->(N, T, C)

        y = self.transducer.infer(
            feats,
            feat_lengths,
            decoding_method=decoding_method,
            beam_width=beam_width,
            max_sym_per_frame=max_sym_per_frame,
            max_sym_per_utt=max_sym_per_utt,
        )
        return y

    def freeze_feat_fuser(self) -> None:
        """Disables gradients on the hidden-state fusion module."""
        if self.feat_fuser is None:
            return

        if self.feat_fusion_method == "weighted-avg":
            self.feat_fuser.requires_grad = False
            return

        for param in self.feat_fuser.parameters():
            param.requires_grad = False

    def freeze_hf_feats(self) -> None:
        """Disables gradients on the Hugging Face frontend."""
        self.hf_feats.freeze()

    def freeze_hf_feature_encoder(self) -> None:
        """Disables gradients on the HF feature encoder submodule."""
        self.hf_feats.freeze_feature_encoder()

    def set_train_mode(self, mode: str) -> None:
        """Updates the wrapper train-mode policy.

        Args:
          mode: Training mode selector.
        """
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        elif mode in ["ft-transducer", "ft-transducer-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
            self.freeze_feat_fuser()
        elif mode in ["hf-feats-frozen", "hf-feats-frozen-nograd"]:
            self.unfreeze()
            self.freeze_hf_feats()
        elif mode == "hf-feat-extractor-frozen":
            self.unfreeze()
            self.freeze_hf_feature_encoder()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        logging.info("train mode set to %s", mode)

        if "nograd" in mode:
            logging.info("using torch.no_grad for hf_feats")
            self._hf_context = torch.no_grad()
        else:
            self._hf_context = contextlib.nullcontext()

        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Internal training-mode switch used by the base class.

        Args:
          train_mode: Training mode selector.
        """

        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        elif train_mode in [
            "ft-transducer",
            "hf-feats-frozen",
            "ft-transducer-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
        ]:
            self.hf_feats.train()
            self.transducer._train("full")
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Returns the supported training modes.

        Returns:
          List of supported training modes.
        """
        return [
            "full",
            "frozen",
            "ft-transducer",
            "hf-feats-frozen",
            "ft-transducer-nograd",
            "hf-feats-frozen-nograd",
            "hf-feat-extractor-frozen",
        ]

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters constructor arguments from a configuration dictionary.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Subset of arguments accepted by this wrapper.
        """
        valid_args = (
            "hf_feats",
            "transducer",
            "feat_fusion_start",
            "feat_fusion_method",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    def get_config(self) -> Dict[str, Any]:
        """Serializes the wrapper configuration.

        Returns:
          Configuration dictionary suitable for reconstruction.
        """
        hf_cfg = self.hf_feats.get_config()
        tran_cfg = self.transducer.get_config()
        del hf_cfg["class_name"]
        del tran_cfg["class_name"]
        config = {
            "hf_feats": hf_cfg,
            "transducer": tran_cfg,
            "feat_fusion_start": self.feat_fusion_start,
            "feat_fusion_method": self.feat_fusion_method,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def change_config(
        self, hf_feats: Dict[str, Any], transducer: Dict[str, Any]
    ) -> None:
        """Applies runtime configuration changes to child modules.

        Args:
          hf_feats: Configuration updates for the Hugging Face frontend.
          transducer: Configuration updates for the backend transducer.
        """
        logging.info("changing hf wav2transducer config")
        self.hf_feats.change_config(**hf_feats)
        self.transducer.change_config(**transducer)

    @staticmethod
    def add_class_args(
        parser: Any, prefix: Optional[str] = None, skip: Optional[Set[str]] = None
    ) -> None:
        """Adds wrapper CLI arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
          skip: Unused compatibility argument.
        """

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--feat-fusion-start",
            default=0,
            type=int,
            help="""
            the input to the transducer will fuse the wav2vec 
            layers from feat_fusion_start to
            the wav2vec num_layers""",
        )
        parser.add_argument(
            "--feat-fusion-method",
            default="weighted-avg",
            choices=["weighted-avg", "linear", "cat", "last"],
            help=(
                "method to fuse the hidden layers from the wav2vec model "
                "in [weighted-avg, linear, cat, last]"
            ),
        )

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    @staticmethod
    def add_infer_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Adds inference CLI arguments to a parser.

        Args:
          parser: Argument parser to extend.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducer.add_infer_args(parser)

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters inference arguments from a configuration dictionary.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Subset of arguments accepted by :meth:`infer`.
        """
        return RNNTransducer.filter_infer_args(**kwargs)

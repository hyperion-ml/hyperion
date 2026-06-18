"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

try:
    import k2
except ModuleNotFoundError:
    from ...utils import dummy_k2 as k2

import torch

from ....utils import HyperDataClass
from ....utils.misc import filter_func_args
from ...narchs import RNNTransducerDecoder
from ...hyper_torch_model import HyperTorchModel


@dataclass
class RNNTransducerOutput(HyperDataClass):
    """Output container for RNN-T training.

    Attributes:
      loss: Total training loss.
      loss_simple: Optional unpruned RNNT loss component.
      loss_pruned: Optional pruned RNNT loss component.
      h_feats: Optional intermediate hidden features.
    """

    loss: torch.Tensor
    loss_simple: Optional[torch.Tensor] = None
    loss_pruned: Optional[torch.Tensor] = None
    h_feats: Optional[List[torch.Tensor]] = None


class RNNTransducer(HyperTorchModel):
    """Base class for RNN-T models.

    Attributes:
      encoder: Optional encoder network module.
      rnnt_decoder: RNN-T decoder module.
    """

    def __init__(
        self,
        encoder: Optional[HyperTorchModel],
        rnnt_decoder: Union[Dict[str, Any], RNNTransducerDecoder],
    ) -> None:
        """Initializes the transducer model.

        Args:
          encoder: Optional encoder network.
          rnnt_decoder: Decoder configuration dictionary or module instance.
        """
        super().__init__()
        if encoder is not None:
            assert isinstance(encoder, HyperTorchModel)
        if isinstance(rnnt_decoder, dict):
            if encoder is not None:
                rnnt_decoder["in_feats"] = encoder.out_shape()[-1]
            rnnt_decoder = RNNTransducerDecoder(**rnnt_decoder)
        else:
            assert isinstance(rnnt_decoder, RNNTransducerDecoder)

        self.encoder = encoder
        self.rnnt_decoder = rnnt_decoder

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: k2.RaggedTensor,
    ) -> RNNTransducerOutput:
        """Computes RNNT training losses.

        Args:
          x: Input features with shape ``(N, T, C)``.
          x_lengths: Number of valid frames for each utterance with shape
            ``(N,)``.
          y: Ragged tensor with axes ``[utt][label]`` containing labels for
            each utterance.

        Returns:
          RNN-T output container with the computed losses.
        """
        assert x.ndim == 3, x.shape
        assert x_lengths.ndim == 1, x_lengths.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lengths.size(0) == y.dim0
        assert torch.all(
            x_lengths[:-1] >= x_lengths[1:]
        ), f"x_lengths={x_lengths}"  # check x_lengths are sorted

        if self.encoder is not None:
            x, x_lengths = self.encoder(x, x_lengths)
            assert torch.all(x_lengths > 0)

        dec_output = self.rnnt_decoder(x, x_lengths, y)
        output = RNNTransducerOutput(*dec_output)
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
        """Decodes a batch of feature sequences.

        Args:
          x: Input features with shape ``(N, T, C)``.
          x_lengths: Number of valid frames for each utterance with shape
            ``(N,)``.
          decoding_method: Decoding algorithm to use.
          beam_width: Beam width for beam search decoders.
          max_sym_per_frame: Maximum number of symbols the RNNT can emit in
            one frame.
          max_sym_per_utt: Maximum number of symbols in a single utterance.

        Returns:
          A list with one decoded token-id sequence per input utterance.
        """
        assert x.ndim == 3, x.shape
        assert x_lengths.ndim == 1, x_lengths.shape
        assert x.size(0) == x_lengths.size(0)

        if self.encoder is not None:
            x, x_lengths = self.encoder(x, x_lengths)
            assert torch.all(x_lengths > 0)

        batch_size = x.size(0)
        y = []
        for i in range(batch_size):
            x_i = x[i : i + 1, : x_lengths[i]]
            y_i = self.rnnt_decoder.decode(
                x_i,
                method=decoding_method,
                beam_width=beam_width,
                max_sym_per_frame=max_sym_per_frame,
                max_sym_per_utt=max_sym_per_utt,
            )
            y.append(y_i)

        return y

    def set_train_mode(self, mode: str) -> None:
        """Updates the model train/eval mode selector.

        Args:
          mode: Either ``"full"`` or ``"frozen"``.
        """
        if mode == self._train_mode:
            return

        if mode == "full":
            self.unfreeze()
        elif mode == "frozen":
            self.freeze()
        else:
            raise ValueError(f"invalid train_mode={mode}")

        self._train_mode = mode

    def _train(self, train_mode: str) -> None:
        """Internal training-mode switch used by the base class.

        Args:
          train_mode: Either ``"full"`` or ``"frozen"``.
        """
        if train_mode in ["full", "frozen"]:
            super()._train(train_mode)
        else:
            raise ValueError(f"invalid train_mode={train_mode}")

    @staticmethod
    def valid_train_modes() -> List[str]:
        """Returns the supported training modes.

        Returns:
          List of supported training modes.
        """
        return ["full", "frozen"]

    def get_config(self) -> Dict[str, Any]:
        """Serializes the model configuration.

        Returns:
          Configuration dictionary suitable for reconstruction.
        """
        if self.encoder is None:
            enc_cfg = None
        else:
            enc_cfg = self.encoder.get_config()
            del enc_cfg["class_name"]

        dec_cfg = self.rnnt_decoder.get_config()
        del dec_cfg["class_name"]
        config = {
            "encoder": enc_cfg,
            "rnnt_decoder": dec_cfg,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters nested configuration dictionaries for construction.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Filtered configuration dictionary.
        """
        args = {}
        rnnt_decoder_args = RNNTransducerDecoder.filter_args(**kwargs["rnnt_decoder"])
        args["rnnt_decoder"] = rnnt_decoder_args
        return args

    @staticmethod
    def add_class_args(
        parser: Any, prefix: Optional[str] = None, skip: Optional[Set[str]] = None
    ) -> None:
        """Registers CLI arguments for this class.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
          skip: Unused compatibility argument.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducerDecoder.add_class_args(parser, prefix="rnnt_decoder")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    def change_config(
        self,
        rnnt_decoder: Dict[str, Any],
    ) -> None:
        """Applies runtime configuration changes to the decoder.

        Args:
          rnnt_decoder: Decoder configuration updates.
        """
        logging.info("changing rnnt_decoder config")
        self.rnnt_decoder.change_config(**rnnt_decoder)

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters fine-tuning configuration dictionaries.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Filtered configuration dictionary.
        """
        args = {}
        rnnt_decoder_args = RNNTransducerDecoder.filter_finetune_args(
            **kwargs["rnnt_decoder"]
        )
        args["rnnt_decoder"] = rnnt_decoder_args
        return args

    @staticmethod
    def add_finetune_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Registers fine-tuning CLI arguments.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        RNNTransducerDecoder.add_finetune_args(parser, prefix="rnnt_decoder")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def add_infer_args(parser: Any, prefix: Optional[str] = None) -> None:
        """Registers inference CLI arguments.

        Args:
          parser: Argument parser where options are registered.
          prefix: Optional namespace prefix for nested parser injection.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--decoding-method",
            default="time_sync_beam_search",
            choices=[
                "greedy",
                "time_sync_beam_search",
                "align_length_sync_beam_search",
            ],
        )

        parser.add_argument(
            "--beam-width", default=5, type=int, help="beam width for beam search"
        )
        parser.add_argument(
            "--max-sym-per-frame",
            default=3,
            type=int,
            help="max symbols RNN-T can emit in 1 frame",
        )
        parser.add_argument(
            "--max-sym-per-utt",
            default=1000,
            type=int,
            help="max symbols RNN-T can emit in 1 frame",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    @staticmethod
    def filter_infer_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters inference configuration dictionaries.

        Args:
          kwargs: Full configuration dictionary.

        Returns:
          Filtered configuration dictionary.
        """
        return filter_func_args(RNNTransducer.infer, kwargs)

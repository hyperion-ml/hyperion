"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...utils.misc import filter_func_args
from ..layer_blocks import TransformerConv2dSubsampler as Subsampler
from .net_arch import NetArch


class RNNEncoder(NetArch):
    """RNN encoder network.

    Attributes:
        in_feats: Input feature dimension.
        hid_feats: Hidden size of the recurrent layers.
        out_feats: Output projection size. When ``0``, the projection layer is
            omitted.
        num_layers: Number of recurrent layers.
        proj_feats: Projection size used by LSTM layers.
        rnn_type: Recurrent cell type, either ``"lstm"`` or ``"gru"``.
        bidirectional: Whether to use bidirectional recurrent layers.
        dropout_rate: Dropout probability used in the recurrent stack and final
            projection.
        subsample_input: Whether to subsample the input time axis by a factor of
            four before the recurrent stack.
        subsampling_act: Activation used by the input subsampler.
        rnn_out_feats: Feature dimension produced by the recurrent stack before
            the final projection.
        _context: Finite-context interface value. RNNs do not have a finite
            receptive field, so this is reported as ``0``.
    """

    def __init__(
        self,
        in_feats: int,
        hid_feats: int,
        out_feats: int,
        num_layers: int,
        proj_feats: int = 0,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout_rate: float = 0.0,
        subsample_input: bool = False,
        subsampling_act: str = "relu",
    ) -> None:
        """Build an RNN encoder.

        Args:
            in_feats: Input feature dimension.
            hid_feats: Hidden size of the recurrent layers.
            out_feats: Output projection size. Use ``0`` to disable the final
                projection layer.
            num_layers: Number of recurrent layers.
            proj_feats: Projection size for LSTM layers.
            rnn_type: Recurrent cell type, either ``"lstm"`` or ``"gru"``.
            bidirectional: Whether to use bidirectional recurrent layers.
            dropout_rate: Dropout probability used in the recurrent stack and
                final projection.
            subsample_input: Whether to subsample the input time axis by a factor
                of four before the recurrent stack.
            subsampling_act: Activation used by the input subsampler.
        """
        super().__init__()
        if rnn_type not in ("lstm", "gru"):
            raise ValueError(f"unsupported rnn_type={rnn_type}")
        if rnn_type != "lstm":
            proj_feats = 0

        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.proj_feats = proj_feats
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.subsample_input = subsample_input
        self.subsampling_act = subsampling_act
        self._context = 0

        self.rnn_out_feats = (
            hid_feats if (rnn_type != "lstm" or proj_feats == 0) else proj_feats
        )
        num_directions = 2 if bidirectional else 1
        self.rnn_out_feats *= num_directions
        if subsample_input:
            self.subsampler = Subsampler(in_feats,
                                         hid_feats,
                                         hid_act=subsampling_act)
            lstm_in_dim = hid_feats
        else:
            self.subsampler = None
            lstm_in_dim = in_feats

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=lstm_in_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                bias=True,
                proj_size=proj_feats,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=lstm_in_dim,
                hidden_size=hid_feats,
                num_layers=num_layers,
                bias=True,
                batch_first=True,
                dropout=dropout_rate,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"unsupported rnn_type={rnn_type}")

        if out_feats > 0:
            self.output = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.rnn_out_feats, out_feats),
            )

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a padded batch of sequences.

        Args:
            x: Input tensor of shape ``(batch, time, feat)``.
            x_lengths: Sequence lengths for each item in ``x``.

        Returns:
            Tuple containing the encoded tensor and the updated sequence lengths.
        """
        if self.subsample_input:
            x, _ = self.subsampler(x)
            x_lengths = (x_lengths + 3) // 4

        x = pack_padded_sequence(input=x,
                                 lengths=x_lengths.cpu(),
                                 batch_first=True,
                                 enforce_sorted=False)
        x, _ = self.rnn(x)
        x, x_lengths = pad_packed_sequence(x, batch_first=True)
        if self.out_feats > 0:
            x = self.output(x)

        return x, x_lengths

    def in_context(self) -> Tuple[int, int]:
        """Return the finite context reported by the encoder.

        Returns:
            Tuple[int, int]: A symmetric ``(left, right)`` context tuple.
        """
        return (self._context, self._context)

    def in_shape(self) -> Tuple[Optional[int], Optional[int], int]:
        """Return the expected input shape.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Expected
            ``(batch, time, feat)`` input shape.
        """
        return (None, None, self.in_feats)

    def out_shape(
        self, in_shape: Optional[Sequence[Optional[int]]] = None
    ) -> Tuple[Optional[int], Optional[int], int]:
        """Return the output shape for an optional input shape.

        Args:
            in_shape: Optional ``(batch, time, feat)`` input shape.

        Returns:
            Tuple[Optional[int], Optional[int], int]: Output
            ``(batch, time, feat)`` shape after the encoder.
        """
        out_feats = self.out_feats if self.out_feats > 0 else self.rnn_out_feats

        if in_shape is None:
            return (None, None, out_feats)

        assert len(in_shape) == 3
        if in_shape[1] is None:
            out_time = None
        elif self.subsample_input:
            out_time = (in_shape[1] + 3) // 4
        else:
            out_time = in_shape[1]

        return (in_shape[0], out_time, out_feats)

    def get_config(self, no_class_name: bool = False) -> Dict[str, Any]:
        """Return a serializable configuration dictionary.

        Args:
            no_class_name: If ``True``, omit the class name from the base
                configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary for reconstructing the
            encoder.
        """
        config = filter_func_args(RNNEncoder.__init__, self.__dict__)
        base_config = super().get_config(no_class_name=no_class_name)
        base_config.update(config)
        return base_config
        #return dict(list(base_config.items()) + list(config.items()))

    def change_config(self, override_dropouts: bool, dropout_rate: float) -> None:
        """Update mutable configuration values.

        Args:
            override_dropouts: Whether to overwrite the dropout probabilities.
            dropout_rate: New dropout probability to apply when requested.
        """
        if override_dropouts:
            logging.info("changing RNNEncoder dropouts")
            self.change_dropouts(dropout_rate)

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter constructor arguments from a keyword dictionary.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Arguments accepted by :meth:`__init__`.
        """
        args = filter_func_args(RNNEncoder.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Set[str] = set(),
    ) -> None:
        """Register constructor arguments with an argument parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used when nesting the encoder options.
            skip: Argument names to omit from the parser.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "in_feats" not in skip:
            parser.add_argument("--in-feats",
                                type=int,
                                required=True,
                                help=("input feature dimension"))

        if "hid_feats" not in skip:
            parser.add_argument(
                "--hid-feats",
                default=1024,
                type=int,
                help=("num of hidden dimensions of RNN layers"),
            )

        if "out_feats" not in skip:
            parser.add_argument(
                "--out-feats",
                default=512,
                type=int,
                help=
                ("number of output dimensions of the encoder, if 0 output projection is removed"
                 ),
            )

        if "proj_feats" not in skip:
            parser.add_argument(
                "--proj-feats",
                default=0,
                type=int,
                help=("projection features of LSTM layers"),
            )

        if "num_layers" not in skip:
            parser.add_argument(
                "--num-layers",
                default=5,
                type=int,
                help=("number of RNN layers"),
            )

        if "rnn_type" not in skip:
            parser.add_argument(
                "--rnn-type",
                default="lstm",
                choices=[
                    "lstm",
                    "gru",
                ],
                help=("RNN type in [lstm, gru]"),
            )

        if "bidirectional" not in skip:
            parser.add_argument(
                "--bidirectional",
                default=False,
                action=ActionYesNo,
                help="whether to use bidirectional RNN",
            )

        if "subsample_input" not in skip:
            parser.add_argument(
                "--subsample-input",
                default=False,
                action=ActionYesNo,
                help="whether to subsaple input features x4",
            )
        if "subsampling_act" not in skip:
            parser.add_argument("--subsampling-act",
                                default="relu",
                                help="activation for subsampler block")

        if "dropout_rate" not in skip:
            parser.add_argument("--dropout-rate",
                                default=0,
                                type=float,
                                help="dropout probability")

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter finetuning-related keyword arguments.

        Args:
            **kwargs: Candidate keyword arguments.

        Returns:
            Dict[str, Any]: Finetuning arguments supported by this class.
        """

        valid_args = (
            "override_dropouts",
            "dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_finetune_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Set[str] = set(),
    ) -> None:
        """Register finetuning arguments with an argument parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional prefix used when nesting the encoder options.
            skip: Argument names to omit from the parser.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "override_dropouts" not in skip:
            try:
                parser.add_argument(
                    "--override-dropouts",
                    default=False,
                    action=ActionYesNo,
                    help=(
                        "whether to use the dropout probabilities passed in the "
                        "arguments instead of the defaults in the pretrained model."
                    ),
                )
            except:
                pass

        if "dropout_rate" not in skip:
            try:
                parser.add_argument("--dropout-rate",
                                    default=0,
                                    type=float,
                                    help="dropout probability")
            except:
                pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))

"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, Optional

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from .etdnn import ETDNNV1
from .net_arch import NetArch
from .resetdnn import ResETDNNV1
from .tdnn import TDNNV1


class TDNNFactory(object):
    """Factory for TDNN family network architectures.

    Attributes:
        None: This is a utility class with only static helpers.
    """

    @staticmethod
    def create(
        tdnn_type: str,
        num_enc_blocks: int,
        in_feats: int,
        enc_hid_units: Any,
        enc_expand_units: Optional[Any] = None,
        kernel_size: Any = 3,
        dilation: Any = 1,
        dilation_factor: int = 1,
        hid_act: Any = {"name": "relu", "inplace": True},
        out_units: int = 0,
        out_act: Any = None,
        dropout_rate: float = 0,
        norm_layer: Optional[str] = None,
        use_norm: bool = True,
        norm_before: bool = True,
        in_norm: bool = True,
    ) -> NetArch:
        """Create a TDNN family network.

        Args:
            tdnn_type: Network type, one of ``"tdnn"``, ``"etdnn"``, or
                ``"resetdnn"``.
            num_enc_blocks: Number of encoder blocks.
            in_feats: Input feature dimension.
            enc_hid_units: Hidden layer width specification.
            enc_expand_units: Final expansion width for ``"resetdnn"``.
            kernel_size: Kernel size or per-block kernel sizes.
            dilation: Dilation or per-block dilations.
            dilation_factor: Dilation increment used when ``dilation`` is a
                scalar.
            hid_act: Hidden activation specification.
            out_units: Output dimension for the final linear layer.
            out_act: Output activation specification.
            dropout_rate: Dropout probability used in the network blocks.
            norm_layer: Normalization layer name.
            use_norm: Whether to enable normalization layers.
            norm_before: Whether normalization happens before activation.
            in_norm: Whether input normalization is enabled.

        Returns:
            NetArch: Instantiated TDNN family module.
        """

        if enc_expand_units is not None and isinstance(enc_hid_units, int):
            if tdnn_type != "resetdnn":
                enc_hid_units = (num_enc_blocks - 1) * [enc_hid_units] + [
                    enc_expand_units
                ]

        if tdnn_type == "tdnn":
            nnet = TDNNV1(
                num_enc_blocks,
                in_feats,
                enc_hid_units,
                out_units=out_units,
                kernel_size=kernel_size,
                dilation=dilation,
                dilation_factor=dilation_factor,
                hid_act=hid_act,
                out_act=out_act,
                dropout_rate=dropout_rate,
                norm_layer=norm_layer,
                use_norm=use_norm,
                norm_before=norm_before,
                in_norm=in_norm,
            )
        elif tdnn_type == "etdnn":
            nnet = ETDNNV1(
                num_enc_blocks,
                in_feats,
                enc_hid_units,
                out_units=out_units,
                kernel_size=kernel_size,
                dilation=dilation,
                dilation_factor=dilation_factor,
                hid_act=hid_act,
                out_act=out_act,
                dropout_rate=dropout_rate,
                norm_layer=norm_layer,
                use_norm=use_norm,
                norm_before=norm_before,
                in_norm=in_norm,
            )
        elif tdnn_type == "resetdnn":
            if enc_expand_units is None:
                if isinstance(enc_hid_units, list):
                    enc_expand_units = enc_hid_units[-1]
                else:
                    enc_expand_units = enc_hid_units

            nnet = ResETDNNV1(
                num_enc_blocks,
                in_feats,
                enc_hid_units,
                enc_expand_units,
                out_units=out_units,
                kernel_size=kernel_size,
                dilation=dilation,
                dilation_factor=dilation_factor,
                hid_act=hid_act,
                out_act=out_act,
                dropout_rate=dropout_rate,
                norm_layer=norm_layer,
                use_norm=use_norm,
                norm_before=norm_before,
                in_norm=in_norm,
            )
        else:
            raise Exception("%s is not valid TDNN network" % (tdnn_type))

        return nnet

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments for TDNN construction.

        This also normalizes legacy aliases such as ``wo_norm`` and
        ``norm_after``.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by :meth:`create`.
        """

        if "wo_norm" in kwargs:
            kwargs["use_norm"] = not kwargs["wo_norm"]
            del kwargs["wo_norm"]

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        valid_args = (
            "tdnn_type",
            "num_enc_blocks",
            "enc_hid_units",
            "enc_expand_units",
            "kernel_size",
            "dilation",
            "dilation_factor",
            "in_norm",
            "hid_act",
            "norm_layer",
            "use_norm",
            "norm_before",
            "in_feats",
            "dropout_rate",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)

        for arg in ("enc_hid_units", "kernel_size", "dilation"):
            if arg in args:
                val = args[arg]
                if isinstance(val, list) and len(val) == 1:
                    args[arg] = val[0]

        return args

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register TDNN constructor arguments on a parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional nested prefix for grouped arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--tdnn-type",
            type=str.lower,
            default="resetdnn",
            choices=["tdnn", "etdnn", "resetdnn"],
            help=("TDNN type: TDNN, ETDNN, ResETDNN"),
        )

        parser.add_argument(
            "--num-enc-blocks",
            default=9,
            type=int,
            help=("number of encoder layer blocks"),
        )

        parser.add_argument(
            "--enc-hid-units",
            nargs="+",
            default=512,
            type=int,
            help=("number of encoder layer blocks"),
        )

        parser.add_argument(
            "--enc-expand-units",
            default=None,
            type=int,
            help=("dimension of last layer of ResETDNN"),
        )

        parser.add_argument(
            "--kernel-size",
            nargs="+",
            default=3,
            type=int,
            help=("kernel sizes of encoder conv1d"),
        )

        parser.add_argument(
            "--dilation",
            nargs="+",
            default=1,
            type=int,
            help=("dilations of encoder conv1d"),
        )

        parser.add_argument(
            "--dilation-factor",
            default=1,
            type=int,
            help=("dilation increment wrt previous conv1d layer"),
        )

        try:
            parser.add_argument("--hid-act", default="relu", help="hidden activation")
        except:
            pass

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
                help="type of normalization layer",
            )
        except:
            pass

        parser.add_argument(
            "--in-norm",
            default=False,
            action="store_true",
            help="batch normalization at the input",
        )

        try:
            parser.add_argument(
                "--wo-norm",
                default=False,
                action="store_true",
                help="without batch normalization",
            )
        except:
            pass

        try:
            parser.add_argument(
                "--norm-after",
                default=False,
                action="store_true",
                help="batch normalizaton after activation",
            )
        except:
            pass

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        try:
            parser.add_argument(
                "--in-feats",
                default=None,
                type=int,
                help=(
                    "input feature dimension, "
                    "if None it will try to infer from encoder network"
                ),
            )
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='TDNN options')

    add_argparse_args = add_class_args

    @staticmethod
    def filter_finetune_args(**kwargs: Any) -> Dict[str, Any]:
        """Filter keyword arguments used during fine-tuning.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: Keyword arguments accepted by
            :meth:`add_finetune_args`.
        """

        valid_args = (
            "override_dropouts",
            "dropout_rate",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_finetune_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Register fine-tuning arguments on a parser.

        Args:
            parser: Argument parser to extend.
            prefix: Optional nested prefix for grouped arguments.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

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

        try:
            parser.add_argument("--dropout-rate", default=0, type=float, help="dropout")
        except:
            pass

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

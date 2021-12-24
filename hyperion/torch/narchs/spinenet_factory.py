"""
 Copyright 2020 Magdalena Rybicka
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from jsonargparse import ArgumentParser, ActionParser

from .spinenet import *

spinenet_dict = {
    "spinenet49": SpineNet49,
    "spinenet49s": SpineNet49S,
    "spinenet96": SpineNet96,
    "spinenet143": SpineNet143,
    "spinenet190": SpineNet190,
    "lspinenet49": LSpineNet49,
    "lspinenet49_subpixel": LSpineNet49_subpixel,
    "lspinenet49_bilinear": LSpineNet49_bilinear,
    "lspinenet49_5": LSpineNet49_5,
    "lspine2net49": LSpine2Net49,
    "selspine2net49": SELSpine2Net49,
    "tselspine2net49": TSELSpine2Net49,
    "spine2net49": Spine2Net49,
    "sespine2net49": SESpine2Net49,
    "tsespine2net49": TSESpine2Net49,
    "spine2net49s": Spine2Net49S,
    "sespine2net49s": SESpine2Net49S,
    "tsespine2net49s": TSESpine2Net49S,
    "lr0_sp53": LR0_SP53,
    "r0_sp53": R0_SP53,
    "spinenet49_concat_time": SpineNet49_concat_time,
}


class SpineNetFactory(object):
    @staticmethod
    def create(
        spinenet_type,
        in_channels,
        output_levels=[3, 4, 5, 6, 7],
        endpoints_num_filters=256,
        resample_alpha=0.5,
        block_repeats=1,
        filter_size_scale=1.0,
        conv_channels=64,
        base_channels=64,
        out_units=0,
        hid_act={"name": "relu6", "inplace": True},
        out_act=None,
        in_kernel_size=7,
        in_stride=2,
        zero_init_residual=False,
        groups=1,
        dropout_rate=0,
        norm_layer=None,
        norm_before=True,
        do_maxpool=True,
        in_norm=True,
        se_r=16,
        in_feats=None,
        res2net_scale=4,
        res2net_width_factor=1,
    ):
        try:
            spinenet_class = spinenet_dict[spinenet_type]
        except:
            raise Exception("%s is not valid SpineNet network" % (spinenet_type))

        spinenet = spinenet_class(
            in_channels,
            output_levels=output_levels,
            endpoints_num_filters=endpoints_num_filters,
            resample_alpha=resample_alpha,
            block_repeats=block_repeats,
            filter_size_scale=filter_size_scale,
            conv_channels=conv_channels,
            base_channels=base_channels,
            out_units=out_units,
            hid_act=hid_act,
            out_act=out_act,
            in_kernel_size=in_kernel_size,
            in_stride=in_stride,
            zero_init_residual=zero_init_residual,
            groups=groups,
            dropout_rate=dropout_rate,
            norm_layer=norm_layer,
            norm_before=norm_before,
            do_maxpool=do_maxpool,
            in_norm=in_norm,
            se_r=se_r,
            in_feats=in_feats,
            res2net_scale=res2net_scale,
            res2net_width_factor=res2net_width_factor,
        )

        return spinenet

    def filter_args(**kwargs):

        if "norm_after" in kwargs:
            kwargs["norm_before"] = not kwargs["norm_after"]
            del kwargs["norm_after"]

        if "no_maxpool" in kwargs:
            kwargs["do_maxpool"] = not kwargs["no_maxpool"]
            del kwargs["no_maxpool"]

        valid_args = (
            "spinenet_type",
            "in_channels",
            "ouput_levels",
            "endpoints_num_filters",
            "resample_alpha",
            "block_repeats",
            "filter_size_scale",
            "conv_channels",
            "base_channels",
            "out_units",
            "hid_act",
            "out_act",
            "in_kernel_size",
            "in_stride",
            "zero_init_residual",
            "groups",
            "dropout_rate",
            "in_norm",
            "norm_layer",
            "norm_before",
            "do_maxpool",
            "se_r",
            "res2net_scale",
            "res2net_width_factor",
            "in_feats",
        )

        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        spinenet_types = spinenet_dict.keys()

        parser.add_argument(
            "--spinenet-type",
            type=str.lower,
            default="spinenet49",
            choices=spinenet_types,
            help=("SpineNet type"),
        )

        parser.add_argument(
            "--in-channels", default=1, type=int, help=("number of input channels")
        )

        parser.add_argument(
            "--conv-channels",
            default=64,
            type=int,
            help=("number of output channels in input convolution "),
        )

        parser.add_argument(
            "--base-channels",
            default=64,
            type=int,
            help=("base channels of first SpineNet block"),
        )

        parser.add_argument(
            "--in-kernel-size",
            default=7,
            type=int,
            help=("kernel size of first convolution"),
        )

        parser.add_argument(
            "--in-stride", default=2, type=int, help=("stride of first convolution")
        )

        parser.add_argument(
            "--groups",
            default=1,
            type=int,
            help=("number of groups in residual blocks convolutions"),
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

        parser.add_argument(
            "--no-maxpool",
            default=False,
            action="store_true",
            help="don't do max pooling after first convolution",
        )

        parser.add_argument(
            "--zero-init-residual",
            default=False,
            action="store_true",
            help="Zero-initialize the last BN in each residual branch",
        )

        parser.add_argument(
            "--se-r",
            default=16,
            type=int,
            help=("squeeze ratio in squeeze-excitation blocks"),
        )

        parser.add_argument(
            "--res2net-scale", default=4, type=int, help=("scale parameter for res2net")
        )

        parser.add_argument(
            "--res2net-width-factor",
            default=1,
            type=float,
            help=("multiplicative factor for the internal width of res2net"),
        )

        try:
            parser.add_argument("--hid-act", default="relu6", help="hidden activation")
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

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

    add_argparse_args = add_class_args

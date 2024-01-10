"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import filter_func_args
from .frplda import FRPLDA
from .plda import PLDA
from .plda_base import PLDALLRNvsMMethod
from .splda import SPLDA


class PLDAType(str, Enum):
    frplda = "frplda"
    splda = "splda"
    plda = "plda"

    @staticmethod
    def choices():
        return [PLDAType.frplda, PLDAType.splda, PLDAType.plda]


class PLDAFactory(object):
    """Class to  create PLDA objects."""

    @staticmethod
    def create(
        plda_type,
        y_dim=None,
        z_dim=None,
        fullcov_W=True,
        update_mu=True,
        update_V=True,
        update_U=True,
        update_B=True,
        update_W=True,
        update_D=True,
        floor_iD=1e-5,
        name="plda",
        **kwargs
    ):
        if plda_type == PLDAType.frplda:
            return FRPLDA(
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_B=update_B,
                update_W=update_W,
                name=name,
                **kwargs
            )
        if plda_type == PLDAType.splda:
            return SPLDA(
                y_dim=y_dim,
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_V=update_V,
                update_W=update_W,
                name=name,
                **kwargs
            )

        if plda_type == PLDAType.plda:
            return PLDA(
                y_dim=y_dim,
                z_dim=z_dim,
                floor_iD=floor_iD,
                update_mu=update_mu,
                update_V=update_V,
                update_U=update_U,
                update_D=update_D,
                name=name,
                **kwargs
            )

    @staticmethod
    def load_plda(plda_type, model_file):
        if plda_type == "frplda":
            return FRPLDA.load(model_file)
        elif plda_type == "splda":
            return SPLDA.load(model_file)
        elif plda_type == "plda":
            return PLDA.load(model_file)

    @staticmethod
    def filter_args(**kwargs):
        return filter_func_args(PLDAFactory.create, kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--plda-type",
            default=PLDAType.splda,
            choices=PLDAType.choices(),
            help="PLDA type",
        )

        parser.add_argument(
            "--y-dim", type=int, default=150, help="num. of eigenvoices"
        )
        parser.add_argument(
            "--z-dim", type=int, default=400, help="num. of eigenchannels"
        )

        parser.add_argument(
            "--fullcov-W",
            default=True,
            action=ActionYesNo,
            help="use full covariance W",
        )
        parser.add_argument(
            "--update-mu",
            default=True,
            action=ActionYesNo,
            help="not update mu",
        )
        parser.add_argument(
            "--update-V", default=True, action=ActionYesNo, help="update V"
        )
        parser.add_argument(
            "--update-U", default=True, action=ActionYesNo, help="update U"
        )

        parser.add_argument(
            "--update-B", default=True, action=ActionYesNo, help="update B"
        )
        parser.add_argument(
            "--update-W", default=True, action=ActionYesNo, help="update W"
        )
        parser.add_argument(
            "--update-D", default=True, action=ActionYesNo, help="update D"
        )
        parser.add_argument(
            "--floor-iD",
            type=float,
            default=1e-5,
            help="floor for inverse of D matrix",
        )

        parser.add_argument("--epochs", type=int, default=40, help="num. of epochs")
        parser.add_argument(
            "--ml-md",
            default="ml+md",
            choices=["ml+md", "ml", "md"],
            help=("optimization type"),
        )

        parser.add_argument(
            "--md-epochs",
            default=None,
            type=int,
            nargs="+",
            help=("epochs in which we do MD, if None we do it in all the epochs"),
        )

        parser.add_argument("--name", default="plda", help="model name")
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    @staticmethod
    def filter_eval_args(**kwargs):
        valid_args = "eval_method"
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_llr_args(parser, prefix=None):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--llr-method", default="vavg", choices=PLDALLRNvsMMethod.choices()
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    # @staticmethod
    # def add_eval_args(parser, prefix=None):
    #     if prefix is None:
    #         p1 = "--"
    #     else:
    #         p1 = "--" + prefix + "."

    #     parser.add_argument(
    #         p1 + "plda-type",
    #         default="splda",
    #         choices=["frplda", "splda", "plda"],
    #         help=("PLDA type"),
    #     )
    #     parser.add_argument(p1 + "model-file", required=True, help=("model file"))

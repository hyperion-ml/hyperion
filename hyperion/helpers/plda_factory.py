"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import numpy as np

from ..np.pdfs.plda import FRPLDA, SPLDA, PLDA


class PLDAFactory(object):
    """Class to  create PLDA objects."""

    @staticmethod
    def create_plda(
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

        if plda_type == "frplda":
            return FRPLDA(
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_B=update_B,
                update_W=update_W,
                name=name,
                **kwargs
            )
        if plda_type == "splda":
            return SPLDA(
                y_dim=y_dim,
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_V=update_V,
                update_W=update_W,
                name=name,
                **kwargs
            )

        if plda_type == "plda":
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
    def filter_train_args(prefix=None, **kwargs):
        valid_args = (
            "plda_type",
            "y_dim",
            "z_dim",
            "diag_W",
            "no_update_mu",
            "no_update_V",
            "no_update_U",
            "no_update_B",
            "no_update_W",
            "no_update_D",
            "floor_iD",
            "epochs",
            "ml_md",
            "md_epochs",
            "name",
        )
        d = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        neg_args1 = (
            "diag_W",
            "no_update_mu",
            "no_update_V",
            "no_update_U",
            "no_update_B",
            "no_update_W",
            "no_update_D",
        )
        neg_args2 = (
            "fullcov_W",
            "update_mu",
            "update_V",
            "update_U",
            "update_B",
            "update_W",
            "update_D",
        )

        for a, b in zip(ne_args1, neg_args2):
            d[b] = not d[a]
            del d[a]

        return d

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "plda-type",
            default="splda",
            choices=["frplda", "splda", "plda"],
            help="PLDA type",
        )

        parser.add_argument(
            p1 + "y-dim", type=int, default=150, help="num. of eigenvoices"
        )
        parser.add_argument(
            p1 + "z-dim", type=int, default=400, help="num. of eigenchannels"
        )

        parser.add_argument(
            p1 + "diag-W",
            default=False,
            action="store_false",
            help="use diagonal covariance W",
        )
        parser.add_argument(
            p1 + "no-update-mu",
            default=False,
            action="store_true",
            help="not update mu",
        )
        parser.add_argument(
            p1 + "no-update-V", default=False, action="store_true", help="not update V"
        )
        parser.add_argument(
            p1 + "no-update-U", default=False, action="store_true", help="not update U"
        )

        parser.add_argument(
            p1 + "no-update-B", default=False, action="store_true", help="not update B"
        )
        parser.add_argument(
            p1 + "no-update-W", default=False, action="store_true", help="not update W"
        )
        parser.add_argument(
            p1 + "no-update-D", default=False, action="store_true", help="not update D"
        )
        parser.add_argument(
            p1 + "floor-iD",
            type=float,
            default=1e-5,
            help="floor for inverse of D matrix",
        )

        parser.add_argument(p1 + "epochs", type=int, default=40, help="num. of epochs")
        parser.add_argument(
            p1 + "ml-md",
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

        parser.add_argument(p1 + "name", default="plda", help="model name")

    @staticmethod
    def filter_eval_args(prefix=None, **kwargs):
        valid_args = ("plda_type", "model_file")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "plda-type",
            default="splda",
            choices=["frplda", "splda", "plda"],
            help=("PLDA type"),
        )
        parser.add_argument(p1 + "model-file", required=True, help=("model file"))

    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args

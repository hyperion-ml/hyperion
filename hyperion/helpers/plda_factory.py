"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ..np.pdfs.plda import FRPLDA, PLDA, SPLDA
from ..utils.misc import PathLike, filter_func_args


class PLDAType(str, Enum):
    frplda = "frplda"
    splda = "splda"
    plda = "plda"

    @staticmethod
    def choices() -> List["PLDAType"]:
        return [PLDAType.frplda, PLDAType.splda, PLDAType.plda]


class PLDAFactory:
    """Class to  create PLDA objects."""

    @staticmethod
    def create(
        plda_type: Union[PLDAType, str],
        y_dim: Optional[int] = None,
        z_dim: Optional[int] = None,
        fullcov_W: bool = True,
        update_mu: bool = True,
        update_V: bool = True,
        update_U: bool = True,
        update_B: bool = True,
        update_W: bool = True,
        update_D: bool = True,
        floor_iD: float = 1e-5,
        prior: Optional[Union[FRPLDA, SPLDA, PLDA, PathLike]] = None,
        r_mu: float = 24.0,
        r_V: float = 128.0,
        r_B: float = 256.0,
        r_W: Optional[float] = None,
        name: str = "plda",
        **kwargs: Any,
    ) -> Union[FRPLDA, SPLDA, PLDA]:
        if prior is not None and isinstance(prior, (str, PathLike)):
            prior = PLDAFactory.load_plda(plda_type, prior)

        if r_W is None:
            r_W = 128.0 if plda_type in (PLDAType.plda, "plda") else 256.0

        if plda_type == PLDAType.frplda:
            return FRPLDA(
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_B=update_B,
                update_W=update_W,
                prior=prior,
                r_mu=r_mu,
                r_B=r_B,
                r_W=r_W,
                name=name,
                **kwargs,
            )
        if plda_type == PLDAType.splda:
            return SPLDA(
                y_dim=y_dim,
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_V=update_V,
                update_W=update_W,
                prior=prior,
                r_mu=r_mu,
                r_V=r_V,
                r_W=r_W,
                name=name,
                **kwargs,
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
                prior=prior,
                r_mu=r_mu,
                r_V=r_V,
                r_W=r_W,
                name=name,
                **kwargs,
            )
        raise ValueError(f"Unsupported PLDA type '{plda_type}'")

    @staticmethod
    def load_plda(
        plda_type: Union[str, PLDAType], model_file: str
    ) -> Union[FRPLDA, SPLDA, PLDA]:
        if plda_type == PLDAType.frplda:
            return FRPLDA.load(model_file)
        elif plda_type == PLDAType.splda:
            return SPLDA.load(model_file)
        elif plda_type == PLDAType.plda:
            return PLDA.load(model_file)
        raise ValueError(f"Unsupported PLDA type '{plda_type}'")

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        return filter_func_args(PLDAFactory.create, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
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
        parser.add_argument(
            "--prior",
            default=None,
            help="prior PLDA model file for Bayesian adaptation",
        )
        parser.add_argument(
            "--r-mu",
            type=float,
            default=24.0,
            help="relevance factor for adapting mu",
        )
        parser.add_argument(
            "--r-V",
            type=float,
            default=128.0,
            help="relevance factor for adapting V",
        )
        parser.add_argument(
            "--r-B",
            type=float,
            default=256.0,
            help="relevance factor for adapting B",
        )
        parser.add_argument(
            "--r-W",
            type=float,
            default=None,
            help=(
                "relevance factor for adapting W "
                "(defaults to 256 for FRPLDA/SPLDA, 128 for PLDA)"
            ),
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
    def filter_eval_args(prefix: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        valid_args = ("plda_type", "model_file")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_eval_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--plda-type",
            default="splda",
            choices=["frplda", "splda", "plda"],
            help=("PLDA type"),
        )
        parser.add_argument("--model-file", required=True, help=("model file"))

        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    add_argparse_train_args = add_class_args
    add_argparse_eval_args = add_eval_args

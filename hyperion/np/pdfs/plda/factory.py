"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ....utils.misc import filter_func_args
from .frplda import FRPLDA
from .plda import PLDA
from .plda_base import PLDALLRNvsMMethod
from .splda import SPLDA


class PLDAType(str, Enum):
    FRPLDA = "frplda"
    SPLDA = "splda"
    PLDA = "plda"

    @staticmethod
    def choices() -> Sequence["PLDAType"]:
        """Returns the valid PLDA back-end types."""
        return [PLDAType.FRPLDA, PLDAType.SPLDA, PLDAType.PLDA]


class PLDAFactory(object):
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
        name: str = "plda",
        **kwargs: Any,
    ) -> Union[FRPLDA, SPLDA, PLDA]:
        """Instantiates a PLDA model using the given configuration.

        Args:
            plda_type: Backend variant to create.
            y_dim: Speaker-factor dimensionality (used by SPLDA/PLDA).
            z_dim: Channel-factor dimensionality (used by PLDA).
            fullcov_W: Whether ``W`` is full covariance for FRPLDA/SPLDA.
            update_mu: Whether ``mu`` is updated during EM.
            update_V: Whether ``V`` is updated (if applicable).
            update_U: Whether ``U`` is updated (PLDA only).
            update_B: Whether ``B`` is updated (FRPLDA).
            update_W: Whether ``W`` is updated.
            update_D: Whether ``D`` is updated (PLDA).
            floor_iD: Minimum inverse variance allowed for ``D``.
            name: Optional model name.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            An initialized PLDA-family instance.
        """
        if plda_type == PLDAType.FRPLDA:
            return FRPLDA(
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_B=update_B,
                update_W=update_W,
                name=name,
                **kwargs,
            )
        if plda_type == PLDAType.SPLDA:
            return SPLDA(
                y_dim=y_dim,
                fullcov_W=fullcov_W,
                update_mu=update_mu,
                update_V=update_V,
                update_W=update_W,
                name=name,
                **kwargs,
            )

        if plda_type == PLDAType.PLDA:
            return PLDA(
                y_dim=y_dim,
                z_dim=z_dim,
                floor_iD=floor_iD,
                update_mu=update_mu,
                update_V=update_V,
                update_U=update_U,
                update_D=update_D,
                name=name,
                **kwargs,
            )
        raise ValueError(f"Unsupported PLDA type '{plda_type}'")

    @staticmethod
    def load_plda(
        plda_type: Union[PLDAType, str], model_file: str
    ) -> Union[FRPLDA, SPLDA, PLDA]:
        """Loads a serialized PLDA model from disk.

        Args:
            plda_type: Type of PLDA stored in ``model_file``.
            model_file: Path to the serialized model.

        Returns:
            Loaded PLDA instance.
        """
        if isinstance(plda_type, str):
            plda_type = PLDAType(plda_type)

        if plda_type == PLDAType.FRPLDA:
            return FRPLDA.load(model_file)
        elif plda_type == PLDAType.SPLDA:
            return SPLDA.load(model_file)
        elif plda_type == PLDAType.PLDA:
            return PLDA.load(model_file)
        raise ValueError(f"Unsupported PLDA type '{plda_type}'")

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments to those accepted by :meth:`create`.

        Args:
            **kwargs: Keyword arguments passed from higher-level configs.

        Returns:
            Dictionary containing only parameters accepted by :meth:`create`.
        """
        return filter_func_args(PLDAFactory.create, kwargs)

    @staticmethod
    def add_class_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds PLDA construction arguments to an :class:`ArgumentParser`.

        Args:
            parser: Target CLI parser.
            prefix: Optional nested prefix for configuration groups.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--plda-type",
            default=PLDAType.SPLDA,
            choices=PLDAType.choices(),
            help="Selects the backend variant (full-rank, simplified, or full PLDA).",
        )

        parser.add_argument(
            "--y-dim",
            type=int,
            default=150,
            help="Latent speaker-factor dimensionality (number of eigenvoices).",
        )
        parser.add_argument(
            "--z-dim",
            type=int,
            default=400,
            help="Latent channel-factor dimensionality (number of eigenchannels).",
        )

        parser.add_argument(
            "--fullcov-W",
            default=True,
            action=ActionYesNo,
            help="Use a full covariance matrix for W instead of a diagonal floor.",
        )
        parser.add_argument(
            "--update-mu",
            default=True,
            action=ActionYesNo,
            help="Enable EM updates of the global mean vector.",
        )
        parser.add_argument(
            "--update-V",
            default=True,
            action=ActionYesNo,
            help="Enable updates of the speaker loading matrix V.",
        )
        parser.add_argument(
            "--update-U",
            default=True,
            action=ActionYesNo,
            help="Enable updates of the channel loading matrix U (PLDA only).",
        )

        parser.add_argument(
            "--update-B",
            default=True,
            action=ActionYesNo,
            help="Enable updates of the between-class precision B (FRPLDA).",
        )
        parser.add_argument(
            "--update-W",
            default=True,
            action=ActionYesNo,
            help="Enable updates of the within-class precision W.",
        )
        parser.add_argument(
            "--update-D",
            default=True,
            action=ActionYesNo,
            help="Enable updates of the diagonal noise precision D (PLDA).",
        )
        parser.add_argument(
            "--floor-iD",
            type=float,
            default=1e-5,
            help="Minimum allowable value for the inverse of each D entry.",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=40,
            help="Number of EM epochs used during training/adaptation.",
        )
        parser.add_argument(
            "--ml-md",
            default="ml+md",
            choices=["ml+md", "ml", "md"],
            help="Training strategy: ML only, MD only, or alternating ML+MD.",
        )

        parser.add_argument(
            "--md-epochs",
            default=None,
            type=int,
            nargs="+",
            help=(
                "Epoch indices where MD updates are applied; when omitted, MD is run "
                "every epoch if enabled."
            ),
        )

        parser.add_argument(
            "--name",
            default="plda",
            help="Optional identifier stored on disk with the trained model.",
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

    @staticmethod
    def filter_eval_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments to those used during evaluation.

        Args:
            **kwargs: Candidate evaluation parameters.

        Returns:
            Dictionary containing only valid evaluation argument names.
        """
        valid_args = "eval_method"
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_llr_args(parser: ArgumentParser, prefix: Optional[str] = None) -> None:
        """Adds LLR scoring arguments to an :class:`ArgumentParser`.

        Args:
            parser: Target CLI parser.
            prefix: Optional nested prefix for configuration groups.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        parser.add_argument(
            "--llr-method",
            default="vavg",
            choices=PLDALLRNvsMMethod.choices(),
            help="Strategy used to pool segments in N-vs-M scoring (book, vavg, etc.).",
        )
        if prefix is not None:
            outer_parser.add_argument(
                "--" + prefix,
                action=ActionParser(parser=parser),
            )

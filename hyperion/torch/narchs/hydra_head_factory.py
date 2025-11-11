"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Type

from jsonargparse import ActionParser, ArgumentParser

from .hydra_heads import HydraClassifHead, HydraHead

_HYDRA_HEAD_REGISTRY: dict[str, Type[HydraHead]] = {
    "classif": HydraClassifHead,
}


class HydraHeadFactory:
    """Factory class for Hydra head modules."""

    DEFAULT_TYPE = "classif"

    @staticmethod
    def supported_types() -> list[str]:
        """Return available head identifiers."""
        return list(_HYDRA_HEAD_REGISTRY.keys())

    @staticmethod
    def create(head_type: str = DEFAULT_TYPE, **kwargs) -> HydraHead:
        """Instantiate a Hydra head of the requested type.

        Args:
            head_type: Identifier of the head to build.
            **kwargs: Keyword arguments forwarded to the concrete head constructor.

        Returns:
            HydraHead: Instantiated hydra head module.
        """
        if head_type is None:
            head_type = HydraHeadFactory.DEFAULT_TYPE

        if head_type not in _HYDRA_HEAD_REGISTRY:
            raise ValueError(
                f"Unsupported head_type '{head_type}'. "
                f"Supported types are: {HydraHeadFactory.supported_types()}"
            )

        head_class = _HYDRA_HEAD_REGISTRY[head_type]
        params = dict(kwargs)
        params.pop("head_type", None)
        params = head_class.filter_args(**params)
        return head_class(**params)

    @staticmethod
    def add_class_args(
        parser: ArgumentParser,
        prefix: Optional[str] = None,
        skip: Optional[set[str]] = None,
    ) -> None:
        """Register CLI/config arguments for Hydra heads.

        Args:
            parser: Target argument parser.
            prefix: Optional nested namespace.
            skip: Optional set of option names to omit.

        """
        if skip is None:
            skip = set()

        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")
        else:
            outer_parser = None

        if "head_type" not in skip:
            parser.add_argument(
                "--head-type",
                choices=HydraHeadFactory.supported_types(),
                default=HydraHeadFactory.DEFAULT_TYPE,
                help="Type of Hydra head to instantiate.",
            )

        # Currently all heads share the same configuration surface, so we expose
        # the arguments of every registered head. If additional heads introduce
        # conflicting options we can scope them under dedicated prefixes.
        HydraHead.add_class_args(parser, prefix=None, skip=skip)
        HydraClassifHead.add_large_margin_loss_args(parser, prefix=None, skip=skip)
        HydraClassifHead.add_label_smoothing_args(parser, prefix=None, skip=skip)

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

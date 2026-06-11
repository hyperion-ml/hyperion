"""
Copyright 2025 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Optional, Type

from jsonargparse import ActionParser, ArgumentParser

from .hydra_heads import (
    HydraClassifHead,
    HydraClassifLossType,
    HydraHead,
    HydraHeadType,
)

_HYDRA_HEAD_REGISTRY: dict[HydraHeadType, Type[HydraHead]] = {
    HydraHeadType.CLASSIF: HydraClassifHead,
}


class HydraHeadFactory:
    """Factory class for Hydra head modules."""

    DEFAULT_TYPE = HydraHeadType.CLASSIF

    @staticmethod
    def supported_types() -> list[str]:
        """Return available head identifiers."""
        return HydraHeadType.choices()

    @staticmethod
    def create(head_type: HydraHeadType = DEFAULT_TYPE, **kwargs) -> HydraHead:
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
    def reconfig_or_create(head: HydraHead, **kwargs) -> HydraHead:
        """Reconfigure an existing head or create a new one if necessary.

        Args:
            head: Existing head instance to reconfigure.
            **kwargs: Keyword arguments forwarded to the head constructor or reconfiguration method.
        Returns:
            HydraHead: Reconfigured or newly created head instance.
        """
        cur_head_type = head.head_type
        new_head_type = kwargs.get("head_type", cur_head_type)
        if new_head_type != cur_head_type:
            cfg = head.get_config(no_class_name=True)
            cfg.update(kwargs)
            return HydraHeadFactory.create(**cfg)
        else:
            if "head_type" in kwargs:
                kwargs.pop("head_type")
            return head.reconfig_or_create(**kwargs)

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
                default=HydraHeadFactory.DEFAULT_TYPE.value,
                help="Type of Hydra head to instantiate.",
            )

        # Currently all heads share the same configuration surface, so we expose
        # the arguments of every registered head. If additional heads introduce
        # conflicting options we can scope them under dedicated prefixes.
        HydraHead.add_class_args(parser, prefix=None, skip=skip)
        if "loss_type" not in skip:
            parser.add_argument(
                "--loss-type",
                default=HydraClassifLossType.ARC_SOFTMAX.value,
                choices=HydraClassifLossType.choices(),
                help="loss type: softmax, arc-softmax, cos-softmax, subcenter-arc-softmax",
            )
        HydraClassifHead.add_large_margin_loss_args(parser, skip=skip)
        HydraClassifHead.add_cross_entropy_loss_args(parser, skip=skip)
        HydraClassifHead.add_prototype_code_rate_args(parser, skip=skip)

        if outer_parser is not None and prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

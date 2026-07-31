"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from dataclasses import dataclass
from typing import Any, ItemsView, KeysView, Type, TypeVar

T = TypeVar("T", bound="HyperDataClass")


@dataclass
class HyperDataClass:
    """Dataclass with dict-like access to attributes.

    This class allows reading/writing dataclass fields through
    ``obj["field_name"]`` in addition to regular attribute access.
    """

    def __getitem__(self, key: str) -> Any:
        """Return the value of an attribute by name.

        Args:
            key: Attribute name.

        Returns:
            Attribute value stored under ``key``.
        """
        return getattr(self, key)

    def __setitem__(self, key: str, val: Any) -> None:
        """Set an attribute value by name.

        Args:
            key: Attribute name.
            val: Value to assign.
        """
        return setattr(self, key, val)

    def keys(self) -> KeysView[str]:
        """Return attribute names."""
        return self.__dict__.keys()

    def items(self) -> ItemsView[str, Any]:
        """Return ``(key, value)`` pairs for attributes."""
        return self.__dict__.items()

    @classmethod
    def from_parent(cls: Type[T], parent: "HyperDataClass", **kwargs: Any) -> T:
        """Create a new instance from an existing instance plus overrides.

        Args:
            parent: Source instance to copy attributes from.
            **kwargs: Field values that override those copied from ``parent``.

        Returns:
            New instance of ``cls`` with merged values.
        """
        args = dict(parent.__dict__)
        args.update(kwargs)
        return cls(**args)


class HypDataClass(HyperDataClass):
    """Backward-compatible alias for HyperDataClass."""

    pass

"""PyTree registration for tensor outputs returned as dataclasses."""

from dataclasses import dataclass, fields
from typing import TypeVar

from torch.utils._pytree import GetAttrKey, register_pytree_node

T = TypeVar("T")


def pytree_dataclass(cls: type[T]) -> type[T]:
    """Create a dataclass whose fields PyTorch can traverse.

    Args:
        cls: Output type whose fields are passed to its dataclass constructor.

    Returns:
        The registered dataclass type.
    """
    cls = dataclass(cls)
    names = tuple(field.name for field in fields(cls))

    def flatten(output):
        return [getattr(output, name) for name in names], None

    def flatten_with_keys(output):
        return [(GetAttrKey(name), getattr(output, name)) for name in names], None

    def unflatten(values, _context):
        return cls(**dict(zip(names, values)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        serialized_type_name=f"{cls.__module__}.{cls.__qualname__}",
        flatten_with_keys_fn=flatten_with_keys,
    )
    return cls

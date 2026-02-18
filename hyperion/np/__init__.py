"""
Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""


from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["HyperNPModel", "NPModel", "NPModelLoader"]

if TYPE_CHECKING:
    from .hyper_np_model import HyperNPModel, NPModel
    from .np_model_loader import NPModelLoader


def __getattr__(name):
    if name == "HyperNPModel":
        return import_module(".hyper_np_model", __name__).HyperNPModel
    if name == "NPModel":
        return import_module(".hyper_np_model", __name__).NPModel
    if name == "NPModelLoader":
        return import_module(".np_model_loader", __name__).NPModelLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

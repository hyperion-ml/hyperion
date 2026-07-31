"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

# from .hyper_torch_model import HyperTorchModel, TorchModel
# from .torch_model_loader import TorchModelLoader

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["HyperTorchModel", "TorchModel", "TorchModelLoader"]

if TYPE_CHECKING:
    from .hyper_torch_model import HyperTorchModel, TorchModel
    from .torch_model_loader import TorchModelLoader


def __getattr__(name):
    if name == "HyperTorchModel":
        return import_module(".hyper_torch_model", __name__).HyperTorchModel
    if name == "TorchModel":
        return import_module(".hyper_torch_model", __name__).TorchModel
    if name == "TorchModelLoader":
        return import_module(".torch_model_loader", __name__).TorchModelLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

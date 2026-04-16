"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import torch.nn as nn


class TorchDataParallel(nn.DataParallel):
    """DataParallel wrapper that forwards missing attributes to ``module``."""

    def __getattr__(self, name):
        """Resolve attributes from this wrapper first, then from wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

"""
Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from torch.optim import RAdam

from .ema import ExpMovingAvg
from .factory import OptimizerFactory
from .fgsm import FGSM

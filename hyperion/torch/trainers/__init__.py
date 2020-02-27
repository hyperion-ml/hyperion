"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

from .torch_trainer import TorchTrainer, TorchDataParallel
from .xvector_trainer import XVectorTrainer
from .xvector_trainer_deep_feat_reg import XVectorTrainerDeepFeatReg
#from .xvector_finetuner import XVectorFinetuner

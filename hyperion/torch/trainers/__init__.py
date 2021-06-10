"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .torch_trainer import TorchTrainer

from .xvector_trainer import XVectorTrainer
from .xvector_trainer_deep_feat_reg import XVectorTrainerDeepFeatReg
from .xvector_adv_trainer import XVectorAdvTrainer

from .xvector_trainer_from_wav import XVectorTrainerFromWav
from .xvector_trainer_deep_feat_reg_from_wav import XVectorTrainerDeepFeatRegFromWav
from .xvector_adv_trainer_from_wav import XVectorAdvTrainerFromWav

from .vae_trainer import VAETrainer
from .dvae_trainer import DVAETrainer
from .vq_vae_trainer import VQVAETrainer
from .vq_dvae_trainer import VQDVAETrainer

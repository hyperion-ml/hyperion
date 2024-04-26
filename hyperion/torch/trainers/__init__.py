"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .dino_xvector_trainer import DINOXVectorTrainer
from .dvae_trainer import DVAETrainer
from .torch_trainer import TorchTrainer
from .transducer_trainer import TransducerTrainer
from .vae_trainer import VAETrainer
from .vq_dvae_trainer import VQDVAETrainer
from .vq_vae_trainer import VQVAETrainer
from .xvector_adv_trainer import XVectorAdvTrainer
from .xvector_adv_trainer_from_wav import XVectorAdvTrainerFromWav
from .xvector_trainer import XVectorTrainer
from .xvector_trainer_deep_feat_reg import XVectorTrainerDeepFeatReg
from .xvector_trainer_deep_feat_reg_from_wav import XVectorTrainerDeepFeatRegFromWav
from .xvector_trainer_from_wav import XVectorTrainerFromWav

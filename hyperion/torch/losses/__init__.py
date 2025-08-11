"""
Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .adv_audio_loss import AudioDiscriminatorAdvLoss, AudioGeneratorAdvLoss
from .bce_with_llr import BCEWithLLR
from .contrastive_loss import ContrastiveLoss
from .convergence_loss import ConvergenceLoss
from .cross_modal_contrastive_loss import CrossModalContrastiveLoss
from .dino_loss import CosineDINOLoss, DINOLoss
from .feature_matching_loss import FeatureMatchingLoss
from .multiresolution_filter_bank_loss import MultiResolutionFilterBankLoss
from .multiresolution_stft_loss import MultiResolutionSTFTLoss
from .si_sdr_loss import SISDRLoss
from .sim_clr_loss import SimCLRLoss

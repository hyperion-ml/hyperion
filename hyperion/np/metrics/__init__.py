"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .utils import effective_prior
from .acc import compute_accuracy
from .confusion_matrix import *
from .eer import compute_eer, compute_prbep
from .dcf import compute_dcf, compute_min_dcf, compute_act_dcf, fast_eval_dcf_eer

"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .acc import compute_accuracy
from .confusion_matrix import *
from .dcf import (compute_act_dcf, compute_dcf, compute_min_dcf,
                  fast_eval_dcf_eer)
from .eer import compute_eer, compute_prbep
from .utils import effective_prior

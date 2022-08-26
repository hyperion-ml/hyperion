"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .devices import open_device
from .metric_acc import MetricAcc
from .masking import seq_lengths_to_mask, scale_seq_lengths
from .collation import collate_seq_1d, collate_seq_2d, collate_seq_nd
from .eval_utils import eval_nnet_by_chunks, eval_nnet_overlap_add
from .vad_utils import remove_silence
from .data_parallel import TorchDataParallel
from .ddp import TorchDDP, FairShardedDDP, FairFullyShardedDDP
from .dinossl import MultiCropWrapper, DINOHead, has_batchnorms, cancel_gradients_last_layer, add_dinossl_args, filter_args, get_params_groups, trunc_normal_, _no_grad_trunc_normal_ 

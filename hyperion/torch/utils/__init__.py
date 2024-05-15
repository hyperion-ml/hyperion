"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .collation import (
    collate_seqs_1d,
    collate_seqs_2d,
    collate_seqs_nd,
    list_of_dicts_to_list,
)
from .data_parallel import TorchDataParallel
from .ddp import FairFullyShardedDDP, FairShardedDDP, TorchDDP
from .devices import (
    open_device,
    tensors_subset,
    tensors_to_cpu,
    tensors_to_device,
    tensors_to_numpy,
)
from .eval_utils import eval_nnet_by_chunks, eval_nnet_overlap_add
from .masking import make_attn_mask_causal, scale_seq_lengths, seq_lengths_to_mask
from .metric_acc import MetricAcc
from .vad_utils import remove_silence

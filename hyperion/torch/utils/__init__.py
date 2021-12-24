"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from .devices import open_device
from .metric_acc import MetricAcc
from .eval_utils import eval_nnet_by_chunks, eval_nnet_overlap_add
from .data_parallel import TorchDataParallel
from .ddp import TorchDDP, FairShardedDDP, FairFullyShardedDDP

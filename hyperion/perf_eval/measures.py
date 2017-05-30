from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from .roc import compute_rocch, rocch2eer


def compute_eer(tar, non):

    p_miss, p_fa = compute_rocch(tar, non)
    return rocch2eer(p_miss, p_fa)


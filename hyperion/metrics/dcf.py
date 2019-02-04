from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

def compute_dcf(p_miss, p_fa, prior, normalize=True):

    dcf = prior * p_miss + (1-prior)* p_fa
    if normalize:
        dcf /= min(prior,1-prior)
    return dcf



    

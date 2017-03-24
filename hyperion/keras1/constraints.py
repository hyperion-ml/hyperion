
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from keras import backend as K
from keras.constraints import Constraint

from ..hyp_defs import float_keras

class ConstraintList(Constraint):

    def __init__(self, constraint_list):
        self.list = constraint_list

    def __call__(self, p):
        for c in self.list:
            p = c(p)
        return p


class Triu(Constraint):

    def __init__(self, dim, diag_val=None):
        self.dim = dim
        self.diag_val = diag_val
        self.diag = None
        if diag_val is None:
            mask = np.triu(np.ones((dim, dim), dtype=float_keras()), 0)
        else:
            mask = np.triu(np.ones((dim, dim), dtype=float_keras()), 1)
            if diag_val != 0:
                #diag = diag_val*np.diag(np.ones((dim,), dtype=float_keras()))
                diag = diag_val*np.eye(dim, dtype=float_keras())
                self.diag = K.variable(diag, dtype=float_keras())
        self.mask = K.variable(mask, dtype=float_keras())
        #print('triu', self.diag, diag_val)
        
    def __call__(self, p):
        p = self.mask*p
        if self.diag is not None:
            p += self.diag
        return p

    
    def get_config(self):
        config = {'dim': self.dim,
                  'diag_val': self.diag_val}
        
        base_config = super(Triu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

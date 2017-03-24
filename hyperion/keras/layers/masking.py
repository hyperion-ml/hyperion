
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer, Merge

class CreateMask(Layer):
    def __init__(self, mask_value=0., **kwargs):
        self.mask_value = mask_value
        super(CreateMask, self).__init__(**kwargs)

    def call(self, x, mask=None):
        boolean_mask = K.any(K.not_equal(x, self.mask_value),
                             axis=-1, keepdims=True)
        if mask is not None:
            boolean_mask*=mask
        return K.cast(boolean_mask, K.floatx())

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(CreateMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

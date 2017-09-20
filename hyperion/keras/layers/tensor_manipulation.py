
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import keras.backend as K
from keras.engine import InputSpec, Layer


class Repeat(Layer):
    '''Repeats the input n times.
       Equivalent to numpy repeat
    '''
    def __init__(self, n, axis, **kwargs):
        super(Repeat, self).__init__(**kwargs)
        self.n = n
        self.axis = axis

        
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[self.axis] is not None:
            output_shape[self.axis] *= self.n
        return tuple(output_shape)


    def call(self, inputs, mask=None):
        return K.repeat_elements(inputs, self.n, axis=self.axis)

    
    def get_config(self):
        config = {'n': self.n,
                  'axis': self.axis }
        base_config = super(Repeat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

    
class ExpandAndTile(Layer):
    '''Expand 1 dimension and tiles
       Equivalent to numpy.tile(numpy.expand)
    '''
    def __init__(self, n, axis=-1, **kwargs):
        super(ExpandAndTile, self).__init__(**kwargs)
        self.n = n
        self.axis = axis
    

    def compute_output_shape(self, input_shape):
        output_shape=list(input_shape)
        output_shape.insert(self.axis,1)
        output_shape[self.axis]*=self.n
        return tuple(output_shape)

    
    def call(self, inputs, mask=None):
        t=K.expand_dims(inputs, axis=self.axis)
        tile_shape=[1]*K.ndims()
        tile_shape[self.axis]=self.n
        return K.tile(t, tuple(tile_shape))

    
    def get_config(self):
        config = {'n': self.n,
                  'axis': self.axis}
        base_config = super(ExpandAndTile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
class ExpandDim(Layer):
    '''Expand 1 dimension
    '''
    def __init__(self, axis=-1, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.axis = axis
    

    def compute_output_shape(self, input_shape):
        output_shape=list(input_shape)
        output_shape.insert(self.axis,1)
        return tuple(output_shape)

    
    def call(self, inputs, mask=None):
        return K.expand_dims(inputs, axis=self.axis)

    
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(ExpandAndTile, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    

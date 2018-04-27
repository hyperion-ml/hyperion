
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import keras.backend as K
from keras.callbacks import Callback


class LearningRateSteps(Callback):
    '''Learning rate scheduler.

    # Arguments
        schedule: a dictionary  (key, val) = (lr change epoch, new_lr)
    '''
    def __init__(self, schedule):
        super(LearningRateSteps, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs={}):
#        assert(hasattr(self.model.optimizer, 'lr'),
#               'Optimizer must have a "lr" attribute.')
        if epoch in self.schedule:
            lr = self.schedule[epoch]
#            print(type(lr))
#            assert(type(lr) == float, 'The output of the "schedule" function should be float.')
            K.set_value(self.model.optimizer.lr, lr)
        


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import logging

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
        

class ReduceLROnPlateauIncreaseOnImprovement(Callback):
    """Reduce learning rate when a metric has stopped improving.
       Increase learning rate if metric is improving

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    # Example

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', red_factor=0.1, inc_factor=1.1, patience=10,
                 mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` insted.')
        self.red_factor = red_factor
        self.inc_factor = inc_factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait_plateau = 0
        self.wait_improv = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

        
    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait_plateau = 0
        self.wait_improv = 0

        
    def on_train_begin(self, logs=None):
        self._reset()

        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait_plateau = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait_plateau = 0
                if not self.in_cooldown():
                    self.wait_improv += 1
                    if self.wait_improv >= self.patience:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                        new_lr = old_lr * self.inc_factor
                        K.set_value(self.model.optimizer.lr, new_lr)
                        logging.info('\nEpoch %05d: IncreaseLROnImprov increasing learning '
                                     'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait_improv = 0
            elif not self.in_cooldown():
                self.wait_plateau += 1
                if self.wait_plateau >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.red_factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        logging.info('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                     'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait_plateau = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

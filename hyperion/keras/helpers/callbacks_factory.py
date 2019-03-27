from __future__ import absolute_import
from __future__ import print_function

import os

from keras.callbacks import *
from ..callbacks import *


class CallbacksFactory(object):

    @staticmethod
    def create_callbacks(model, file_path, save_all_epochs=False, 
                         monitor = 'val_loss',
                         monitor_mode='auto', patience=None, min_delta=1e-4,
                         lr_steps = None,
                         lr_monitor = None, lr_monitor_mode='auto',
                         lr_patience = None, lr_factor=0.1,
                         lr_red_factor=None, lr_inc_factor=None,
                         min_lr=1e-7,
                         log_append=False):

        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        
        if lr_monitor is None:
            lr_monitor = monitor
        
        if save_all_epochs:
            file_path_model = file_path + '/model.{epoch:04d}'        
        else:
            file_path_model = file_path + '/model.best'
        cb = HypModelCheckpoint(model, file_path_model, monitor=monitor, verbose=1,
                                save_best_only=not(save_all_epochs),
                                save_weights_only=False, mode=monitor_mode)
        cbs = [cb]

        file_path_csv = file_path + '/train.log'
        cb = CSVLogger(file_path_csv, separator=',', append=log_append)
        cbs.append(cb)
    
        if patience is not None:
            cb = EarlyStopping(monitor=monitor, patience=patience,
                               min_delta=min_delta, verbose=1, mode=monitor_mode)
            cbs.append(cb)
        
        if lr_steps is not None:
            cb = LearningRateSteps(lr_steps)
            cbs.append(cb)    

        if lr_patience is not None:
            if lr_inc_factor is None:
                if lr_red_factor is not None:
                    lr_factor = lr_red_factor
                cb = ReduceLROnPlateau(monitor=lr_monitor,
                                       factor=lr_factor, patience=lr_patience,
                                       verbose=1, mode=lr_monitor_mode, min_delta=min_delta,
                                       cooldown=0, min_lr=min_lr)
            else:
                cb = ReduceLROnPlateauIncreaseOnImprovement(
                    monitor=lr_monitor,
                    red_factor=lr_red_factor, inc_factor=lr_inc_factor,
                    patience=lr_patience,
                    verbose=1, mode=lr_monitor_mode, min_delta=min_delta,
                    cooldown=0, min_lr=min_lr)
                
            cbs.append(cb)    
        
        return cbs


    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('save_all_epochs', 'monitor_mode', 'lr_monitor_mode',
                      'monitor', 'patience', 'min_delta',
                      'lr_monitor',
                      'lr_steps', 'lr_patience', 'lr_factor',
                      'lr_red_factor', 'lr_inc_factor',
                      'min_lr', 'log_append')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)


    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        
        parser.add_argument(p1+'monitor', dest=(p2+'monitor'), 
                            default='val_loss')
        parser.add_argument(p1+'lr-monitor', dest=(p2+'lr_monitor'), 
                            default=None)

        parser.add_argument(p1+'monitor-mode', dest=(p2+'monitor_mode'), 
                            default='auto',choices=['min','max','auto'])
        parser.add_argument(p1+'lr-monitor-mode', dest=(p2+'lr_monitor_mode'), 
                            default='auto',choices=['min','max','auto'])
        
        parser.add_argument(p1+'patience', dest=(p2+'patience'), default=100,
                            type=int,
                            help=('Training stops after PATIENCE epochs without '
                                  'improvement of the validation loss '
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'lr-patience', dest=(p2+'lr_patience'), default=10,
                            type=int,
                            help=('Multiply the learning rate by LR_FACTOR '
                                  'after LR_PATIENCE epochs without '
                                  'improvement of the validation loss '
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'lr-factor', dest=(p2+'lr_factor'), default=0.1,
                            type=float,
                            help=('Learning rate scaling factor '
                                  '(default: %(default)s)'))
        
        parser.add_argument(p1+'lr-red-factor', dest=(p2+'lr_red_factor'), default=None,
                            type=float,
                            help=('Scaling factor to reduce learning rate'
                                  '(default: %(default)s)'))
        
        parser.add_argument(p1+'lr-inc-factor', dest=(p2+'lr_inc_factor'), default=None,
                            type=float,
                            help=('Scaling factor to increase learning rate'
                                  '(default: %(default)s)'))

        parser.add_argument(p1+'min-delta', dest=(p2+'min_delta'), default=1e-4,
                            type=float,
                            help=('Minimum improvement'
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'min-lr', dest=(p2+'min_lr'), default=1e-5,
                            type=float,
                            help=('Minimum learning rate'
                                  '(default: %(default)s)'))
        parser.add_argument(p1+'lr-steps', dest=(p2+'lr_steps'), nargs='+',
                            default=None)
        parser.add_argument(p1+'save-all-epochs', dest=(p2+'save_all_epochs'),
                            default=False, action='store_true')
        parser.add_argument(p1+'log-append', dest=(p2+'log_append'),
                            default=False, action='store_true')
        

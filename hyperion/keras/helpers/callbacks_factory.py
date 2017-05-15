from __future__ import absolute_import
from __future__ import print_function

from keras.callbacks import *
from ..callbacks import *


class CallbacksFactory(object):

    @staticmethod
    def create_callbacks(model, file_path, save_best_only=True, mode='min',
                         monitor = 'val_loss', patience=None, min_delta=1e-4,
                         lr_steps = None,
                         lr_patience = None, lr_factor=0.1, min_lr=1e-5,
                         log_append=False):

        if save_best_only == True:
            file_path_model = file_path + '/model.best'
        else:
            file_path_model = file_path + '/model.{epoch:04d}'
        cb = HypModelCheckpoint(model, file_path_model, monitor=monitor, verbose=1,
                                save_best_only=save_best_only,
                                save_weights_only=False, mode=mode)
        cbs = [cb]

        file_path_csv = file_path + '/train.log'
        cb = CSVLogger(file_path_csv, separator=',', append=log_append)
        cbs.append(cb)
    
        if patience is not None:
            cb = EarlyStopping(monitor=monitor, patience=patience,
                               min_delta=min_delta, verbose=1, mode=mode)
            cbs.append(cb)
        
        if lr_steps is not None:
            cb = LearningRateSteps(lr_steps)
            cbs.append(cb)    

        if lr_patience is not None:
            cb = ReduceLROnPlateau(monitor=monitor,
                                   factor=lr_factor, patience=lr_patience,
                                   verbose=1, mode=mode, epsilon=min_delta,
                                   cooldown=0, min_lr=min_lr)
            cbs.append(cb)    
        
        return cbs


    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('save_best_only', 'mode',
                      'monitor', 'patience', 'min_delta',
                      'lr_steps', 'lr_patience', 'lr_factor',
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
        parser.add_argument(p1+'save-all-epochs', dest=(p2+'save_best_only'),
                            default=True, action='store_false')
        

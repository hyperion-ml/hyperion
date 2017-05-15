from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from ..pdfs.plda import FRPLDA, SPLDA

class PLDAFactory(object):

    @staticmethod
    def create_plda(plda_type, y_dim=None, z_dim=None, fullcov_W=True,
                    update_mu=True, update_V=True, update_B=True, update_W=True,
                    name='plda',
                    **kwargs):
        if plda_type == 'frplda':
            return FRPLDA(fullcov_W=fullcov_W,
                          update_mu=update_mu, update_B=update_B,
                          update_W=update_W, name=name, **kwargs)
        if plda_type == 'splda':
            return SPLDA(y_dim=y_dim, fullcov_W=fullcov_W,
                         update_mu=update_mu, update_V=update_V,
                         update_W=update_W, name=name, **kwargs)


        
    @staticmethod
    def load_plda(plda_type, model_file):
        if plda_type == 'frplda':
            return FRPLDA.load(model_file)
        elif plda_type == 'splda':
            return SPLDA.load(model_file)



    @staticmethod
    def filter_train_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('plda_type', 'y_dim', 'z_dim',
                      'fullcov_W', 'update_mu', 'update_V', 'update_B', 'update_W',
                      'epochs', 'ml_md', 'md_epochs', 'name')
        return dict((k, kwargs[p+k])
                    for k in valid if p+k in kwargs)


    
        
    @staticmethod
    def add_argparse_train_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'plda-type', dest=(p2+'plda_type'), 
                            default='splda',
                            choices=['frplda', 'splda'],
                            help='PLDA type')
        
        parser.add_argument(p1+'y-dim', dest=(p2+'y_dim'), type=int,
                            default=150,
                            help='num. of eigenvoices')
        parser.add_argument(p1+'z-dim', dest=(p2+'z_dim'), type=int,
                            default=400,
                            help='num. of eigenchannels')

        parser.add_argument(p1+'fullcov-W', dest=(p2+'fullcov_W'), type=bool,
                            default=True,
                            help='use full covariance W')
        parser.add_argument(p1+'update-mu', dest=(p2+'update_mu'), type=bool,
                            default=True,
                            help='update mu')
        parser.add_argument(p1+'update-V', dest=(p2+'update_V'), type=bool,
                            default=True,
                            help='update V')
        parser.add_argument(p1+'update-B', dest=(p2+'update_B'), type=bool,
                            default=True,
                            help='update B')
        parser.add_argument(p1+'update-W', dest=(p2+'update_W'), type=bool,
                            default=True,
                            help='update W')

        parser.add_argument(p1+'epochs', dest=(p2+'epochs'), type=int,
                            default=40,
                            help='num. of epochs')
        parser.add_argument(p1+'ml-md', dest=(p2+'ml_md'), 
                            default='ml+md',
                            choices=['ml+md', 'ml', 'md'],
                            help=('optimization type'))

        parser.add_argument('--md-epochs', dest='md_epochs', default=None,
                            type=int, nargs = '+',
                            help=('epochs in which we do MD, if None we do it in all the epochs'))

        parser.add_argument(p1+'name', dest=(p2+'name'), 
                            default='plda',
                            help='model name')

        

    @staticmethod
    def filter_eval_args(prefix, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('plda_type', 'model_file')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        
    @staticmethod
    def add_argparse_eval_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'plda-type', dest=(p2+'plda_type'), 
                            default='splda',
                            choices=['frplda', 'splda'],
                            help=('PLDA type'))
        parser.add_argument(p1+'model-file', dest=(p2+'model_file'), required=True,
                            help=('model file'))
                            
        
        

"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import numpy as np

from ..pdfs.plda import FRPLDA, SPLDA, PLDA

class PLDAFactory(object):
    """Class to  create PLDA objects."""
    
    @staticmethod
    def create_plda(plda_type, y_dim=None, z_dim=None, fullcov_W=True,
                    update_mu=True, update_V=True, update_U=True,
                    update_B=True, update_W=True, update_D=True,
                    floor_iD=1e-5,
                    name='plda', **kwargs):
        
        if plda_type == 'frplda':
            return FRPLDA(fullcov_W=fullcov_W,
                          update_mu=update_mu, update_B=update_B,
                          update_W=update_W, name=name, **kwargs)
        if plda_type == 'splda':
            return SPLDA(y_dim=y_dim, fullcov_W=fullcov_W,
                         update_mu=update_mu, update_V=update_V,
                         update_W=update_W, name=name, **kwargs)

        if plda_type == 'plda':
            return PLDA(y_dim=y_dim, z_dim=z_dim, floor_iD=floor_iD,
                        update_mu=update_mu, update_V=update_V,
                        update_U=update_U, update_D=update_D,
                        name=name, **kwargs)


        
    @staticmethod
    def load_plda(plda_type, model_file):
        if plda_type == 'frplda':
            return FRPLDA.load(model_file)
        elif plda_type == 'splda':
            return SPLDA.load(model_file)
        elif plda_type == 'plda':
            return PLDA.load(model_file)


    @staticmethod
    def filter_train_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('plda_type', 'y_dim', 'z_dim',
                      'diag_W', 'no_update_mu', 'no_update_V', 'no_update_U',
                      'no_update_B', 'no_update_W', 'no_update_D', 'floor_iD',
                      'epochs', 'ml_md', 'md_epochs', 'name')
        d = dict((k, kwargs[p+k])
                 for k in valid_args if p+k in kwargs)
        neg_args1 = ('diag_W', 'no_update_mu', 'no_update_V', 'no_update_U',
                      'no_update_B', 'no_update_W', 'no_update_D')
        neg_args2 = ('fullcov_W', 'update_mu', 'update_V', 'update_U',
                      'update_B', 'update_W', 'update_D')

        for a,b in zip(ne_args1, neg_args2):
            d[b] = not d[a]
            del d[a]

        return d

    
        
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
                            choices=['frplda', 'splda', 'plda'],
                            help='PLDA type')
        
        parser.add_argument(p1+'y-dim', dest=(p2+'y_dim'), type=int,
                            default=150,
                            help='num. of eigenvoices')
        parser.add_argument(p1+'z-dim', dest=(p2+'z_dim'), type=int,
                            default=400,
                            help='num. of eigenchannels')

        parser.add_argument(p1+'diag-W', dest=(p2+'diag_W'),
                            default=False, action='store_false',
                            help='use diagonal covariance W')
        parser.add_argument(p1+'no-update-mu', dest=(p2+'no_update_mu'),
                            default=False, action='store_true',
                            help='not update mu')
        parser.add_argument(p1+'no-update-V', dest=(p2+'no_update_V'),
                            default=False, action='store_true',
                            help='not update V')
        parser.add_argument(p1+'no-update-U', dest=(p2+'no_update_U'),
                            default=False, action='store_true',
                            help='not update U')

        parser.add_argument(p1+'no-update-B', dest=(p2+'no_update_B'),
                            default=False, action='store_true',
                            help='not update B')
        parser.add_argument(p1+'no-update-w', dest=(p2+'no_update_W'),
                            default=False, action='store_true',
                            help='not update W')
        parser.add_argument(p1+'no-update-d', dest=(p2+'no_update_D'),
                            default=False, action='store_true',
                            help='not update D')
        parser.add_argument(p1+'floor-id', dest=(p2+'floor_iD'), type=float,
                            default=1e-5,
                            help='floor for inverse of D matrix')

        
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
                            choices=['frplda', 'splda', 'plda'],
                            help=('PLDA type'))
        parser.add_argument(p1+'model-file', dest=(p2+'model_file'), required=True,
                            help=('model file'))
                            
        
        

"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import numpy as np
import h5py

import scipy.linalg as la

from ..hyp_model import HypModel


class PCA(HypModel):
    """Class to do principal component analysis
    """
    def __init__(self, mu=None, T=None, update_mu=True, update_T=True, 
                 pca_dim=None, pca_var_r=None, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.T = T
        self.update_mu = update_mu
        self.update_T = update_T
        self.pca_dim = pca_dim
        self.pca_var_r = pca_var_r

        
    def predict(self, x):
        if self.mu is not None:
            x = x - self.mu
        return np.dot(x, self.T)


    
    def fit(self, x=None, sample_weight=None, mu=None, C=None):
        
        if x is not None:
            mu = np.mean(x, axis=0)
            delta = x - mu
            C = np.dot(delta.T, delta)/x.shape[0]

        if self.update_mu:
            self.mu = mu

        if self.update_T:
            d, V = la.eigh(C)
            V = np.fliplr(V)

            p = V[0,:] < 0
            V[:,p] *= -1

            if self.pca_var_r is not None:
                d = np.flip(d)
                var_acc = np.cumsum(d)
                var_r = var_acc/var_acc[-1]
                self.pca_dim = np.where(var_r > self.pca_var_r)[0][0]
        
            if self.pca_dim is not None:
                assert self.pca_dim <= V.shape[1]
                V = V[:,:self.pca_dim]

            self.T = V


            
    def get_config(self):
        config = {'update_mu': self.update_mu,
                  'update_t': self.update_T,
                  'pca_dim': self.pca_dim,
                  'pca_var_r': self.pca_var_r}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    
        
    def save_params(self, f):
        params = {'mu': self.mu,
                  'T': self.T}
        self._save_params_from_dict(f, params)

        

    @classmethod
    def load_params(cls, f, config):
        param_list = ['mu', 'T']
        params = cls._load_params_to_dict(f, config['name'], param_list)
        return cls(mu=params['mu'], T=params['T'], pca_dim=config['pca_dim'], name=config['name'])
    

    
    @classmethod
    def load_mat(cls, file_path):
        with h5py.File(file_path, 'r') as f:
            mu = np.asarray(f['mu'], dtype='float32')
            T = np.asarray(f['T'], dtype='float32')
            return cls(mu, T)


        
    def save_mat(self, file_path):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('mu', data=self.mu)
            f.create_dataset('T', data=self.T)



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
            
        valid_args = ('update_mu', 'update_T', 'name', 'pca_dim', 'pca_var_r')
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
            
        parser.add_argument(p1+'update-mu', default=True, type=bool,
                            help=('updates centering parameter'))
        parser.add_argument(p1+'update-t', dest=(p2+'update_T'), default=True,
                            type=bool,
                            help=('updates whitening parameter'))

        parser.add_argument(p1+'pca-dim', default=None, type=int,
                            help=('output dimension of PCA'))

        parser.add_argument(p1+'pca-var-r', default=None, type=int,
                            help=('proportion of variance to keep when choosing the PCA dimension'))

        parser.add_argument('--name', dest='name', default='pca')

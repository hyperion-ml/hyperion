"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange


import numpy as np
from ..hyp_defs import float_cpu


def MeanVarianceNorm(object):
    """Class to perform mean and variance normalization
    
    Attributes:
       norm_mean: normalize mean
       norm_var: normalize variance
       left_context: past context of the sliding window, if None all past frames.
       right_context: future context of the sliding window, if None all future frames.
    
    If left_context==right_context==None, it will apply global mean/variance normalization.
    """
    def __init__(self, norm_mean=True, norm_var=False, left_context=None, right_context=None):
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.left_context = left_context
        self.right_context = right_context


    def normalize(self, x):
        """Normalize featurex in x
        
        Args:
          x: Input feature matrix.

        Returns:
          Normalized feature matrix.
        """


        # Global mean/var norm.
        if self.norm_mean:
            m_x = np.mean(x, axis=0, keepdims=True)
            x = x - m_x

        if self.norm_var:
            s_x = np.std(x, axis=0, keepdims=True)
            x = x/s_x

        if self.right_context is None and self.left_context is None:
            return x

        if self.left_context is None:
            left_context = x.shape[0]
        else:
            left_context = self.left_context

        if self.right_context is None:
            right_context = x.shape[0]
        else
            right_context = self.right_context

        total_context = left_context + right_contest + 1

        if x.shape[0] <= min(right_context, left_context)+1:
            # if context is larger than the signal we still return global normalization
            return x

        c_x = np.zeros((x.shape[0]+total_context+1, x.shape[1],), dtype=float_cpu())
        counts = np.zeros((x.shape[0]+total_context+1, 1,), dtype=float_cpu())
        
        c_x[left_context:left_context+x.shape[0]] = np.cumsum(x, axis=0)
        c_x[left_context+x.shape[0]:] = c_x[left_context+x.shape[0]-1]
        counts[left_context:left_context+x.shape[0]] = np.arange(1, x.shape[0]+1, dtype=float_cpu())
        counts[left_context+x.shape[0]:] = x.shape[0]

        if self.norm_var == True:
            c2_x[left_context:left_context+x.shape[0]] = np.cumsum(x*x, axis=0)
            c2_x[left_context+x.shape[0]:] = c2_x[left_context+x.shape[0]-1]
        
        m_x = (c_x[total_context:] - c_x[:-total_context])/counts

        if self.norm_mean:
            x -= m_x

        if self.norm_var:
            m2_x = (c2_x[total_context:] - c2_x[:-total_context])/counts
            s_x = np.sqrt(m2_x - m_x**2)
            s_x[s_x<1e-5] = 1e-5
            x /= s_x

        return x
        # m_x = np.zeros_like(x)
        # if self.norm_var:
        #     c2_x = np.cumsum(x*x, axis=0)
        #     m2_x = np.zeros_like(x)

            
            
        # # short-time mean/var norm.
        # if x.shape[0] > total_context - 1:
        #     # When signal is larger than context

        #     # For frames 0 to left_context
        #     denom = np.arange(1, left_context+2, dtype=float_cpu())[:,None] + self.right_context
        #     m_x[:left_context+1] = c_x[right_context:total_context]/denom
        #     if self.norm_var:
        #         m2_x = c2_x[rigth_context:total_context]/denom

        #     # For frames left_context + 1 to total_frames - right_context - 1
        #     if x.shape[0] > total_context:
        #         denom = total_context
        #         m_x[left_context+1:-right_context] = (c_x[total_context:] - c_x[:-total_context])/denom
        #         if self.norm_var:
        #             m2_x[left_context+1:-right_context] = (c_x[total_context:] - c2_x[:-total_context])/denom

        #     # For frames total_frames - right_context to the end
        #     denom = np.arange(right_context, 0, -1,  dtype=float_cpu())[:,None] + self.left_context
        #     m_x[-right_context:] = - c_x[-total_context:-left_context] + c_x[/denom
        #     if self


    def normalize_slow(self, x):

        # Global mean/var norm.
        if self.norm_mean:
            m_x = np.mean(x, axis=0, keepdims=True)
            x = x - m_x

        if self.norm_var:
            s_x = np.std(x, axis=0, keepdims=True)
            x = x/s_x

        if self.right_context is None and self.left_context is None:
            return x
        
        m_x = np.zeros_like(x)
        s_x = np.zeros_like(x)

        for i in xrange(x.shape[0]):
            idx1 = max(i-left_context, 0)
            idx2 = min(i+right_context, x.shape[0]-1) + 1
            denom = idx2 - idx1
            m_x[i] = np.mean(x[idx1:idx2], axis=0)
            s_x[i] = np.std(x[idx1:idx2], axis=0)


        if self.norm_mean:
            x -= m_x
        if self.norm_var:
            s_x[s_x<1e-5] = 1e-5
            x /= s_x

        return x



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        """Filters ST-CMVN args from arguments dictionary.
           
           Args:
             prefix: Options prefix.
             kwargs: Arguments dictionary.
           
           Returns:
             Dictionary with ST-CMVN options.
        """
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('no_norm_mean', 'norm_mean', 'norm_var', 'left_context', 'right_context')

        d = dict((k, kwargs[p+k])
                 for k in valid_args if p+k in kwargs)

        neg_args1 = ('no_norm_mean')
        neg_args2 = ('norm_mean')

        for a,b in zip(ne_args1, neg_args2):
            d[b] = not d[a]
            del d[a]

        return d

    
        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        """Adds ST-CMVN options to parser.
           
           Args:
             parser: Arguments parser
             prefix: Options prefix.
        """

        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'

        parser.add_argument(p1+'no-norm-mean', dest=(p2+'no_norm_mean'), 
                            default=False, action='store_true',
                            help='don\'t center the features')

        parser.add_argument(p1+'norm-var', dest=(p2+'norm_var'), 
                            default=False, action='store_true',
                            help='normalize the variance of the features')

        
        parser.add_argument(p1+'left-context', dest=(p2+'left_context'), type=int,
                            default=300,
                            help='past context in number of frames')

        parser.add_argument(p1+'right-context', dest=(p2+'right_context'), type=int,
                            default=300,
                            help='future context in number of frames')

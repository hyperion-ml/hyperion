"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import OrderedDict as ODict
import numpy as np


class MetricAcc(object):
    """Class to accumulate metrics during an epoch.
    """
    def __init__(self):
        self.keys = None
        self.acc = None
        self.count = 0


    def reset(self):
        """Resets the accumulators.
        """
        self.count = 0
        if self.acc is not None:
            self.acc[:] = 0
            
        
    def update(self, metrics, num_samples=1):
        """Updates the values of the metric

           It uses recursive formula, it may be more numerically stable

               m^(i) = m^(i-1) + n^(i)/sum(n^(i)) (x^(i) - m^(i-1))
               
              where i is the batch number, 
              m^(i) is the accumulated average of the metric at batch i,
              x^(i) is the average of the metric at batch i,
              n^(i) is the batch_size at batch i.
              
           Args:
               metrics: dictionary with metrics for current batch
               num_samples: number of samples in current batch (batch_size)
        """
        if self.keys is None:
            self.keys = metrics.keys()
            self.acc = np.zeros((len(self.keys),))

        self.count += num_samples
        r = num_samples/self.count
        for i, k in enumerate(self.keys):
            self.acc[i] += r * (metrics[k] - self.acc[i])



    @property
    def metrics(self):
        """ Returns metrics dictionary
        """
        logs = ODict()
        for i,k in enumerate(self.keys):
            logs[k] = self.acc[i]

        return logs
                                
        

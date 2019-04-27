"""
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

class LoggerList(object):
    """Container for a list of logger callbacks

    Attributes:
       loggers: list of Logger objects
    """
    def __init__(self, loggers=None):
        self.loggers = loggers or []

        
    def append(self, logger):
        self.loggers.append(logger)

        
    def on_epoch_begin(self, epoch, logs=None, **kwargs):
        """At the start of an epoch
        
        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_begin(epoch, logs, **kwargs)

            
    def on_epoch_end(self, logs=None, **kwargs):
        """At the end of an epoch
        
        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_epoch_end(logs, **kwargs)


    def on_batch_begin(self, batch, logs=None, **kwargs):
        """At the start of a batch
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_begin(batch, logs, **kwargs)


    def on_batch_end(self, logs=None, **kwargs):
        """At the end of a batch
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_batch_end(logs, **kwargs)


    def on_train_begin(self, logs=None, **kwargs):
        """At the start of training
        
        Args:
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_begin(logs, **kwargs)


    def on_train_end(self, logs=None, **kwargs):
        """At the end of training
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        logs = logs or {}
        for logger in self.loggers:
            logger.on_train_end(logs, **kwargs)


    def __iter__(self):
        return iter(self.loggers)

    


class Logger(object):
    """Base class for logger objects
    
    Attributes:
       params: training params dictionary
    """
    def __init__(self):
        self.cur_epoch = 0
        self.cur_batch = 0
        self.params=None

    
    def on_epoch_begin(self, epoch, logs, **kwargs):
        """At the start of an epoch
        
        Args:
           epoch: index of the epoch
           logs: dictionary of logs
        """
        self.cur_epoch = epoch
        
            
    def on_epoch_end(self, logs, **kwargs):
        """At the end of an epoch
        
        Args:
           logs: dictionary of logs
        """
        pass

    
    def on_batch_begin(self, batch, logs, **kwargs):
        """At the start of a batch
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        self.cur_batch = batch
        

    def on_batch_end(self, logs, **kwargs):
        """At the end of a batch
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        pass

    def on_train_begin(self, logs, **kwargs):
        """At the start of training
        
        Args:
           logs: dictionary of logs
        """
        pass

    def on_train_end(self, logs, **kwargs):
        """At the end of training
        
        Args:
           batch: batch index within the epoch
           logs: dictionary of logs
        """
        pass



    

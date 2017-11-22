"""
Base class to write ark or h5 files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

from abc import ABCMeta, abstractmethod


class DataWriter(object):
    __metaclass__ = ABCMeta

    def __init__(self, archive_path, script_path=None,
                 flush=False, compress=False, compression_method='auto'):
        self.archive_path = archive_path
        self.script_path = script_path
        self.flush = flush
        self.compress = compress
        self.compression_method = compression_method
        

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def flush(self):
        pass
    
    @abstractmethod
    def write(self, key, data):
        pass

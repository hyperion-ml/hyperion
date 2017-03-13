"""
Class to read input and target features from hdf5 files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import numpy as np
import h5py

from ..utils.list_utils import list2ndarray, ismember

class HypDataReader(object):

    def __init__(self, file_path, load_to_ram=True):
        self.file_path = file_path
        self.load_to_ram = load_to_ram

        self.f = h5py.File(file_path, 'r')

        self.key = list2ndarray([t.decode('utf-8') for t in f['key']])
        self.perm = None

        
    def nb_seqs(self):
        return len(self.key)

    
    def shuffle_seqs(self, rng):
        if isinstance(rng, np.random.mtrand.RandomState):
            self.perm = rng.permutation(self.nb_seqs())
        elif isinstance(rng,np.ndarray):
            self.perm = rng
        else:
            raise TypeError

        
    def get_seqs_by_idx(self, idx, dataset):
        if self.perm is not None:
            idx = self.perm[idx]
        assert(dataset in self.f, '%f has not %s' % (self.file_path, dataset))
        dssl = dataset + '/seq_lengths'
        dsss = dataset + '/start_seq'
        dsd =  dataset + '/data'
        assert(dsd in self.f, '%f has not %s' % (self.file_path, dsd))
        assert(dssl in self.f, '%f has not %s' % (self.file_path, dssl))
        assert(dsss in self.f, '%f has not %s' % (self.file_path, dsss))

        seq_start = f[dsss][idx]
        seq_length = f[dssl][idx]
        total_seq_length = np.sum(seq_length)
        dim = f[dsd].shape[1]
        data = np.zeros((total_seq_length, dim), dtype='float32')
        j=0
        for i in xrange(len(idx)):
            s_i = seq_start[i]
            l_i = seq_length[i]
            data[j:j+l_i,:] = f[dsd][s_i:s_i+l_i,:]
            j +=l_i
        return data

    
    def get_segs_by_key(self, key, dataset):
        if self.perm is not None:
            self.perm = None
        f, idx = ismember(key, self.key)
        assert(np.all(f))
        return self.get_seq_by_idx(idx, dataset)

    
    def total_seq_length(self, dataset):
        assert(dataset in self.f, '%f has not %s' % (self.file_path, dataset))
        dssl = dataset + '/seq_lengths'
        assert(dssl in self.f, '%f has not %s' % (self.file_path, dssl))
        return np.sum(self.f[dssl])

    
    def dim(self, dataset):
        assert(dataset in self.f, '%f has not %s' % (self.file_path, dataset))
        dsd =  dataset + '/data'
        assert(dsd in self.f, '%f has not %s' % (self.file_path, dsd))

        return f[dsd].shape[1]
        
        
        
        
         
        

                
            
                
    

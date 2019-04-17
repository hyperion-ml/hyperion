"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import os
import copy
import numpy as np
from numpy.testing import assert_allclose

from hyperion.utils import Utt2Info
from hyperion.io import H5DataWriter
from hyperion.generators.sequence_batch_generator_v2 import SequenceBatchGeneratorV2 as SBG

output_dir = './tests/data_out/generators'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

h5_file = output_dir + '/seqbgv2.h5'
key_file = output_dir + '/seqbgv2.scp'

num_seqs = 10
dim = 2
min_seq_length = 100
delta = 10
max_seq_length = min_seq_length + (num_seqs-1)*delta
seq_lengths = np.arange(100, max_seq_length+1, delta)


def create_dataset():

    file_path = [str(k) for k in xrange(num_seqs)]
    key=[]
    i = 0
    j = 0
    while i < num_seqs:
        key_i = (j+1)*str(j)
        i += (i+1)
        j += 1
        key += key_i
    key = key[:num_seqs]

    u2c = Utt2Info.create(file_path, key)

    if os.path.exists(h5_file):
        return u2c
    
    u2c.save(key_file, sep=' ')

    h = H5DataWriter(h5_file)
    rng = np.random.RandomState(seed=0)

    for i in xrange(num_seqs):
        x_i = rng.randn(seq_lengths[i], dim)
        h.write(file_path[i], x_i)
    
    return u2c



def test_num_seqs():

    create_dataset()
    sr = SBG(h5_file, key_file)
    assert sr.num_seqs==num_seqs


    
def test_seq_lengths():
    
     create_dataset()
     sr = SBG(h5_file, key_file, shuffle_seqs=False)
     
     assert np.all(sr.seq_lengths==seq_lengths)
     assert sr.total_length==np.sum(seq_lengths)
     assert sr.min_seq_length == min_seq_length
     assert sr.max_seq_length == max_seq_length


     
def test_num_total_subseqs():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5)
    sr.num_total_subseqs == num_seqs


    
def test_prune_min_length():
    
    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, shuffle_seqs=False,
             prune_min_length=min_seq_length+5)
    assert sr.num_seqs==num_seqs - 1
    assert np.all(sr.seq_lengths==seq_lengths[1:])
    assert sr.total_length==np.sum(seq_lengths[1:])
    assert sr.min_seq_length == np.min(seq_lengths[1:])
    assert sr.max_seq_length == max_seq_length



def test_class_info():
    
    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, shuffle_seqs=False)
    assert sr.num_classes == 4

    print(sr.u2c.info)
    print(sr.u2c.key)
    print(sr.class2utt)
    print(sr.class2num_utt)
    class2utt = {0: ['0'], 1: ['1','2'], 2: ['3','4','5'], 3: ['6','7','8','9']}
    class2utt_idx = {0: np.array([0], dtype=int),
                     1: np.array([1,2], dtype=int),
                     2: np.array([3,4,5], dtype=int),
                     3: np.array([6,7,8,9], dtype=int)}
    #class2num_utt = {0: 1, 1: 2, 2: 3, 3: 4}
    class2num_utt = np.array([1,2,3,4], dtype=int)
    assert sr.class2utt == class2utt
    assert np.all(sr.class2num_utt == class2num_utt)
    for k in sr.class2utt.keys():
        assert_allclose(sr.class2utt_idx[k], class2utt_idx[k])

    
def test_compute_iters_auto():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5)
    assert sr.iters_per_epoch == 1

    sr = SBG(h5_file, key_file, batch_size=5,
             max_seq_length=min_seq_length)
    assert sr.iters_per_epoch == 2


    
def test_reset():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, reset_rng=True,
             min_seq_length=5, max_seq_length=17)

    x1 = sr.rng.randn(3)
    sr.reset()
    assert sr.cur_step == 0
    assert_allclose(x1, sr.rng.randn(3))


    sr = SBG(h5_file, key_file, batch_size=5, reset_rng=False,
             min_seq_length=5, max_seq_length=17)
    
    sr.cur_epoch = 100
    sr.reset()
    assert sr.cur_step == 0
    assert np.mean(x1) != np.mean(sr.rng.rand(3))
    

        
def read_func(batch_size, nepc, nepu):

    u2c = create_dataset()
    sr = SBG(h5_file, key_file, 
             reset_rng=True, iters_per_epoch=2,
             num_egs_per_class=nepc,
             num_egs_per_utt=nepu,
             min_seq_length=10, max_seq_length=20,
             batch_size=batch_size)

    x_e = []
    for epoch in xrange(2):
        x0 = x_e
        key_e = []
        c_e = []
        x_e = []
        sw_e = []
        for i in xrange(sr.steps_per_epoch):
            key_i, x_i, sw_i, y_i = sr.read()
            c_i =[ i for i in np.argmax(y_i, axis=-1)]
            assert len(x_i) == batch_size
            key_e += key_i
            c_e += c_i
            x_e.append(x_i)
            sw_e.append(sw_i)
            
            skey, key_ids = np.unique(key_i, return_inverse=True)
            counts_key = np.zeros((len(skey),), dtype=int)
            for k in xrange(len(skey)):
                counts_key[k] = np.sum(key_ids==k)
            assert np.all(counts_key>=nepu) and np.all(counts_key<=batch_size)

            sc, c_ids = np.unique(c_i, return_inverse=True)
            counts_c = np.zeros((len(sc),), dtype=int)
            for k in xrange(len(sc)):
                counts_c[k] = np.sum(c_ids==k)
            assert np.all(counts_c>=nepc*nepu) and np.all(counts_c<=batch_size)
            
        x_e = np.vstack(tuple(x_e))
        sw_e = np.vstack(tuple(sw_e))
        sl_e = np.sum(sw_e, axis=-1).astype(int)

        if epoch > 0:
            assert_allclose(x0, x_e)
        assert np.all(np.logical_and(sl_e>=10, sl_e<=20))
        print(c_e)
        print(key_e)
        # assert 0


def test_read_1epc_1epu():
    read_func(5,1,1)


def test_read_2epc_1epu():
    read_func(6,2,1)


def test_read_2epc_2epu():
    read_func(16,2,2)

        
if __name__ == '__main__':
    pytest.main([__file__])




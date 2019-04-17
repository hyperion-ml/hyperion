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
from hyperion.generators.sequence_batch_generator_v1 import SequenceBatchGeneratorV1 as SBG

output_dir = './tests/data_out/generators'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

h5_file = output_dir + '/seqbg.h5'
key_file = output_dir + '/seqbg.scp'

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
    assert sr.num_seqs == num_seqs


    
def test_seq_lengths():
    
     create_dataset()
     sr = SBG(h5_file, key_file, shuffle_seqs=False)
     
     assert np.all(sr.seq_lengths == seq_lengths)
     assert sr.total_length==np.sum(seq_lengths)
     assert sr.min_seq_length == min_seq_length
     assert sr.max_seq_length == max_seq_length


     
def test_num_total_subseqs():

    create_dataset()
    sr = SBG(h5_file, key_file, gen_method='full_seqs', batch_size=5)
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

    print(sr.u2c.key)
    print(sr.u2c.info)
    class_ids = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    key2class = { p:k for p,k in zip(sr.u2c.key, class_ids)}
    assert sr.key2class == key2class


    
def test_balance_class_weight():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, class_weight='unbalanced', shuffle_seqs=False)
    u2c0 = sr.u2c
    
    sr = SBG(h5_file, key_file, batch_size=5, class_weight='balanced', shuffle_seqs=False, max_class_imbalance=1)

    class_ids = [0]*4 + [1]*4 + [2]*4 + [3]*4
    class_ids = [ str(i) for i in class_ids]
    key = [u2c0.key[0]]*4 + list(u2c0.key[1:3])*2 + list(
        u2c0.key[3:6]) + [u2c0.key[3]] + list(u2c0.key[6:])
    print(key)
    print(class_ids)
    print(sr.u2c.key)
    print(sr.u2c.info)
    u2c = Utt2Info.create(key, class_ids)
    assert u2c == sr.u2c
    


def test_compute_iters_auto():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, gen_method='random', shuffle_seqs=False)
    assert sr.iters_per_epoch == 1

    sr = SBG(h5_file, key_file, batch_size=5, gen_method='random', shuffle_seqs=False,
             max_seq_length=min_seq_length)
    assert sr.iters_per_epoch == 2


    
def test_prepare_sequential_subseqs():
    
    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, gen_method='sequential',
             shuffle_seqs=False,
             min_seq_length=5, max_seq_length=17, seq_overlap=1)
    print(sr._init_num_subseqs)
    assert_allclose(sr._init_num_subseqs, seq_lengths/10)


    
def test_reset():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, gen_method='sequential',
             shuffle_seqs=False,
             min_seq_length=5, max_seq_length=17, seq_overlap=1)

    u2c = sr.init_u2c
    seq_lengths = sr.seq_lengths
    num_subseqs = sr._init_num_subseqs
    
    sr.shuffle_seqs = True
    sr.reset()

    assert u2c == sr.init_u2c
    assert_allclose(seq_lengths, sr._init_seq_lengths)
    assert_allclose(num_subseqs, sr._init_num_subseqs)
    
    idx1 = np.argsort(u2c.key)
    idx2 = np.argsort(sr.u2c.key)

    u2c1 = u2c.filter_index(idx1)
    u2c2 = sr.u2c.filter_index(idx2)

    assert u2c1 == u2c2
    assert_allclose(seq_lengths[idx1], sr.seq_lengths[idx2])
    assert_allclose(num_subseqs[idx1], sr.num_subseqs[idx2])
    assert np.all(sr.cur_subseq == 0)
    assert np.all(sr.cur_frame == 0)


    
def test_read_full_seq():

    u2c = create_dataset()
    sr = SBG(h5_file, key_file, shuffle_seqs=False,
             gen_method='full_seqs', batch_size=5)

    x_e = []
    for epoch in xrange(2):
        x0 = x_e
        key_e = []
        c_e = []
        x_e = []
        sw_e = []
        for i in xrange(sr.steps_per_epoch):
            key_i, x_i, sw_i, y_i = sr.read()
            assert len(x_i) == 5
            key_e += key_i
            c_e += [ str(i) for i in np.argmax(y_i, axis=-1)]
            x_e.append(x_i)
            sw_e.append(sw_i)
        x_e = np.vstack(tuple(x_e))
        sw_e = np.vstack(tuple(sw_e))
        sl_e = np.sum(sw_e, axis=-1).astype(int)

        if epoch > 0:
            assert_allclose(x0, x_e)
        assert_allclose(seq_lengths, sl_e)
        u2c_e = Utt2Info.create(key_e, c_e)
        assert u2c == u2c_e


        
def test_read_random():

    u2c = create_dataset()
    u2c = Utt2Info.merge([u2c]*2)
    sr = SBG(h5_file, key_file, shuffle_seqs=False,
             reset_rng=True, iters_per_epoch=2,
             min_seq_length=10, max_seq_length=20,
             gen_method='random', batch_size=5)

    x_e = []
    for epoch in xrange(2):
        x0 = x_e
        key_e = []
        c_e = []
        x_e = []
        sw_e = []
        for i in xrange(sr.steps_per_epoch):
            key_i, x_i, sw_i, y_i = sr.read()
            assert len(x_i) == 5
            key_e += key_i
            c_e += [ str(i) for i in np.argmax(y_i, axis=-1)]
            x_e.append(x_i)
            sw_e.append(sw_i)
        x_e = np.vstack(tuple(x_e))
        sw_e = np.vstack(tuple(sw_e))
        sl_e = np.sum(sw_e, axis=-1).astype(int)

        if epoch > 0:
            assert_allclose(x0, x_e)
        assert np.all(np.logical_and(sl_e>=10, sl_e<=20))
        u2c_e = Utt2Info.create(key_e, c_e)
        assert u2c == u2c_e



def test_read_sequential_balanced():

    u2c = create_dataset()
    u2c = Utt2Info.merge([u2c]*int(np.max(seq_lengths)/10))
    sr = SBG(h5_file, key_file, shuffle_seqs=False,
             reset_rng=True, 
             min_seq_length=5, max_seq_length=17, seq_overlap=1,
             gen_method='sequential', seq_weight='balanced',
             batch_size=5)

    x_e = []
    for epoch in xrange(2):
        x0 = x_e
        key_e = []
        c_e = []
        x_e = []
        sw_e = []
        for i in xrange(sr.steps_per_epoch):
            key_i, x_i, sw_i, y_i = sr.read()
            assert len(x_i) == 5
            key_e += key_i
            c_e += [ str(i) for i in np.argmax(y_i, axis=-1)]
            x_e.append(x_i)
            sw_e.append(sw_i)
        x_e = np.vstack(tuple(x_e))
        sw_e = np.vstack(tuple(sw_e))
        sl_e = np.sum(sw_e, axis=-1).astype(int)

        if epoch > 0:
            assert_allclose(x0, x_e)
        assert np.all(np.logical_and(sl_e>=5, sl_e<=17))
        # print(u2c.key)
        # print(u2c.info)
        # print(np.array(key_e))
        # print(np.array(c_e))

        u2c_e = Utt2Info.create(key_e, c_e)
        assert u2c == u2c_e



def test_read_sequential_unbalanced():

    u2c = create_dataset()
    u2c_list = [u2c]*int(np.min(seq_lengths)/10)
    for i in xrange(1,num_seqs):
        u2c_list.append(u2c.filter_index(
            np.arange(i,num_seqs)))
    u2c = Utt2Info.merge(u2c_list)
    sr = SBG(h5_file, key_file, shuffle_seqs=False,
             reset_rng=True, 
             min_seq_length=5, max_seq_length=17, seq_overlap=1,
             gen_method='sequential', seq_weight='unbalanced',
             batch_size=5)

    x_e = []
    for epoch in xrange(2):
        x0 = x_e
        key_e = []
        c_e = []
        x_e = []
        sw_e = []
        for i in xrange(sr.steps_per_epoch):
            key_i, x_i, sw_i, y_i = sr.read()
            assert len(x_i) == 5
            key_e += key_i
            c_e += [ str(i) for i in np.argmax(y_i, axis=-1)]
            x_e.append(x_i)
            sw_e.append(sw_i)
        x_e = np.vstack(tuple(x_e))
        sw_e = np.vstack(tuple(sw_e))
        sl_e = np.sum(sw_e, axis=-1).astype(int)

        if epoch > 0:
            assert_allclose(x0, x_e)
        assert np.all(np.logical_and(sl_e>=5, sl_e<=17))
        print(u2c.key)
        print(u2c.info)
        print(np.array(key_e))
        print(np.array(c_e))

        u2c_e = Utt2Info.create(key_e, c_e)
        assert u2c == u2c_e

        
    
if __name__ == '__main__':
    pytest.main([__file__])





from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import os
import copy
import numpy as np
from numpy.testing import assert_allclose

from hyperion.utils.scp_list import SCPList
from hyperion.io import H5DataWriter
from hyperion.helpers.sequence_batch_generator import SequenceBatchGenerator as SBG

h5_file = './tests/data_out/seqbg.h5'
key_file = './tests/data_out/seqbg.scp'

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

    scp = SCPList(key, file_path)

    if os.path.exists(h5_file):
        return scp
    
    scp.save(key_file, sep=' ')

    h = H5DataWriter(h5_file)
    rng = np.random.RandomState(seed=0)

    for i in xrange(num_seqs):
        x_i = rng.randn(seq_lengths[i], dim)
        h.write(file_path[i], x_i)
    
    return scp



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

    print(sr.scp.key)
    print(sr.scp.file_path)
    class_ids = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    key2class = { p:k for p,k in zip(sr.scp.file_path, class_ids)}
    assert sr.key2class == key2class


    
def test_balance_class_weight():

    create_dataset()
    sr = SBG(h5_file, key_file, batch_size=5, class_weight='unbalanced', shuffle_seqs=False)
    scp0 = sr.scp
    
    sr = SBG(h5_file, key_file, batch_size=5, class_weight='balanced', shuffle_seqs=False)

    key = [0]*4 + [1]*4 + [2]*4 + [3]*4
    key = [ str(i) for i in key]
    file_path = [scp0.file_path[0]]*4 + list(scp0.file_path[1:3])*2 + list(
        scp0.file_path[3:6]) + [scp0.file_path[3]] + list(scp0.file_path[6:])
    print(key)
    print(file_path)
    print(sr.scp.key)
    print(sr.scp.file_path)
    scp = SCPList(key, file_path)
    assert scp == sr.scp
    


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

    scp = sr.init_scp
    seq_lengths = sr.seq_lengths
    num_subseqs = sr._init_num_subseqs
    
    sr.shuffle_seqs = True
    sr.reset()

    assert scp == sr.init_scp
    assert_allclose(seq_lengths, sr._init_seq_lengths)
    assert_allclose(num_subseqs, sr._init_num_subseqs)
    
    idx1 = np.argsort(scp.file_path)
    idx2 = np.argsort(sr.scp.file_path)

    scp1 = scp.filter_index(idx1)
    scp2 = sr.scp.filter_index(idx2)

    assert scp1 == scp2
    assert_allclose(seq_lengths[idx1], sr.seq_lengths[idx2])
    assert_allclose(num_subseqs[idx1], sr.num_subseqs[idx2])
    assert np.all(sr.cur_subseq == 0)
    assert np.all(sr.cur_frame == 0)


    
def test_read_full_seq():

    scp = create_dataset()
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
        scp_e = SCPList(c_e, key_e)
        assert scp == scp_e


        
def test_read_random():

    scp = create_dataset()
    scp = SCPList.merge([scp]*2)
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
        scp_e = SCPList(c_e, key_e)
        assert scp == scp_e



def test_read_sequential_balanced():

    scp = create_dataset()
    scp = SCPList.merge([scp]*int(np.max(seq_lengths)/10))
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
        # print(scp.key)
        # print(scp.file_path)
        # print(np.array(c_e))
        # print(np.array(key_e))

        scp_e = SCPList(c_e, key_e)
        assert scp == scp_e



def test_read_sequential_unbalanced():

    scp = create_dataset()
    scp_list = [scp]*int(np.min(seq_lengths)/10)
    for i in xrange(1,num_seqs):
        scp_list.append(scp.filter_index(
            np.arange(i,num_seqs)))
    scp = SCPList.merge(scp_list)
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
        print(scp.key)
        print(scp.file_path)
        print(np.array(c_e))
        print(np.array(key_e))

        scp_e = SCPList(c_e, key_e)
        assert scp == scp_e



# def test_read_sequential():
    

#     sr = SBG(h5_file, key_file, shuffle_seqs=False, batch_size=5,
#                         max_seq_length=20,
#                         seq_split_mode='sequential', seq_split_overlap=5)

#     #read epoch 1
#     x1=[]
#     for i in xrange(sr.num_batches):
#         x1_i = sr.read()[0]
#         assert(len(x1_i)==5)
#         x1 += x1_i

#     #read epoch 2
#     x2=[]
#     for i in xrange(sr.num_batches):
#         x2_i = sr.read()[0]
#         assert(len(x2_i)==5)
#         x2 += x2_i

#     assert(len(x1)==sr.num_total_subseqs)
#     assert(len(x1)==len(x2))
#     for i in xrange(len(x1)):
#         assert(x1[i].shape[0]<=sr.max_batch_seq_length)
#         assert(np.all(x1[i]==x2[i]))



# def test_read_random():
    

#     sr = SBG(h5_file, key_file, shuffle_seqs=False, batch_size=5,
#              max_seq_length=20, min_seq_length=20,
#              seq_split_mode='random_slice_1seq')
#     print(sr.num_batches)
#     #read epoch 1
#     x1=[]
#     for i in xrange(sr.num_batches):
#         x1_i = sr.read()[0]
#         assert(len(x1_i)==5)
#         x1 += x1_i

#     #read epoch 2
#     x2=[]
#     for i in xrange(sr.num_batches):
#         x2_i = sr.read()[0]
#         assert(len(x2_i)==5)
#         x2 += x2_i

#     assert(int(len(x1)/5)==sr.num_batches)
#     assert(len(x1)==sr.num_seqs)
#     assert(len(x1)==sr.num_total_subseqs)
#     assert(len(x1)==len(x2))
#     for i in xrange(len(x1)):
#         assert(x1[i].shape[0]==sr.max_batch_seq_length)
#         assert(x2[i].shape[0]==sr.max_batch_seq_length)
#         assert(np.all(x1[i]!=x2[i]))


        
    
if __name__ == '__main__':
    pytest.main([__file__])




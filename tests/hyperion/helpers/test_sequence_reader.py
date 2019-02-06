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

from hyperion.utils.scp_list import SCPList
from hyperion.io import HypDataWriter
from hyperion.helpers import SequenceReader

output_dir = './tests/data_out/helpers'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

h5_file = output_dir + '/seqr.h5'
key_file = output_dir + '/seqr.scp'

num_seqs = 10
dim = 2
min_seq_length = 100
delta = 10
max_seq_length = min_seq_length + (num_seqs-1)*delta
seq_length = np.arange(100, max_seq_length+1, delta)


def create_dataset():

    if os.path.exists(h5_file):
        return

    file_path = [str(k) for k in xrange(num_seqs)]
    scp = SCPList(file_path, file_path)
    scp.save(key_file, sep='=')

    h = HypDataWriter(h5_file)
    rng = np.random.RandomState(seed=0)

    for i in xrange(num_seqs):
        x_i = rng.randn(seq_length[i], dim)
        h.write(file_path[i], '', x_i)
    

def test_num_seqs():

    create_dataset()
    sr = SequenceReader(h5_file, key_file)
    assert(sr.num_seqs==num_seqs)


def test_seq_length():
    
    create_dataset()
    sr = SequenceReader(h5_file, key_file)
    assert(np.all(sr.seq_length==seq_length))
    assert(sr.total_length==np.sum(seq_length))
    

def test_num_subseqs():

    create_dataset()
    sr = SequenceReader(h5_file, key_file, min_seq_length=min_seq_length+1)
    num_subseqs_gt = np.ones((num_seqs,), dtype=int)
    num_subseqs_gt[0] = 0
    assert(np.all(sr.num_subseqs==num_subseqs_gt))

    sr = SequenceReader(h5_file, key_file, max_seq_length=delta*2)
    num_subseqs_gt = np.array([5, 6, 6, 7, 7, 8, 8, 9, 9, 10], dtype=int)
    assert(np.all(sr.num_subseqs==num_subseqs_gt))

    sr = SequenceReader(h5_file, key_file,
                        max_seq_length=delta*2, min_seq_length=delta*2)
    num_subseqs_gt = np.array([5, 5, 6, 6, 7, 7, 8, 8, 9, 9], dtype=int)
    assert(np.all(sr.num_subseqs==num_subseqs_gt))

    sr = SequenceReader(h5_file, key_file,
                        max_seq_length=delta*2, min_seq_length=delta*2,
                        seq_split_overlap=delta/2)

    num_subseqs_gt = np.array([6, 7, 7, 8, 9, 9, 10, 11, 11, 12], dtype=int)
    print(sr.num_subseqs)
    assert(np.all(sr.num_subseqs==num_subseqs_gt))
    
    

def test_max_batch_seq_length():
    
    create_dataset()
    sr = SequenceReader(h5_file, key_file)
    assert(sr.max_batch_seq_length==max_seq_length)

    sr = SequenceReader(h5_file, key_file, max_seq_length=min_seq_length/4)
    assert(sr.max_batch_seq_length==min_seq_length/4)


def test_num_batches():

    create_dataset()
    sr = SequenceReader(h5_file, key_file, batch_size=5,
                        max_seq_length=delta*2, min_seq_length=delta*2,
                        seq_split_overlap=delta/2)
    print(sr.num_batches)
    assert(sr.num_batches==18)



def test_reset():

    create_dataset()
    sr = SequenceReader(h5_file, key_file,
                        max_seq_length=delta*2, min_seq_length=delta*2,
                        seq_split_overlap=delta/2)
    scp = copy.deepcopy(sr.scp)
    seq_length = sr.seq_length
    num_subseqs = sr.num_subseqs
    sr.reset()
    assert(scp != sr.scp)
    assert(not np.all(seq_length == sr.seq_length))
    assert(not np.all(num_subseqs == sr.num_subseqs))


def test_read_full_seq():

    create_dataset()


    sr = SequenceReader(h5_file, key_file, shuffle_seqs=False, batch_size=5)

    seq_length=sr.seq_length
    
    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]==seq_length[i])
        assert(np.all(x1[i]==x2[i]))
        


def test_read_sequential():
    

    sr = SequenceReader(h5_file, key_file, shuffle_seqs=False, batch_size=5,
                        max_seq_length=20,
                        seq_split_mode='sequential', seq_split_overlap=5)

    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(len(x1)==sr.num_total_subseqs)
    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]<=sr.max_batch_seq_length)
        assert(np.all(x1[i]==x2[i]))



def test_read_random_slice_1seq():
    

    sr = SequenceReader(h5_file, key_file, shuffle_seqs=False, batch_size=5,
                        max_seq_length=20, min_seq_length=20,
                        seq_split_mode='random_slice_1seq')
    print(sr.num_batches)
    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(int(len(x1)/5)==sr.num_batches)
    assert(len(x1)==sr.num_seqs)
    assert(len(x1)==sr.num_total_subseqs)
    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]==sr.max_batch_seq_length)
        assert(x2[i].shape[0]==sr.max_batch_seq_length)
        assert(np.all(x1[i]!=x2[i]))




def test_read_random_samples_1seq():
    

    sr = SequenceReader(h5_file, key_file, shuffle_seqs=False, batch_size=5,
                        max_seq_length=20, min_seq_length=20,
                        seq_split_mode='random_samples_1seq')

    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(int(len(x1)/5)==sr.num_batches)
    assert(len(x1)==sr.num_seqs)
    assert(len(x1)==sr.num_total_subseqs)
    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]==sr.max_batch_seq_length)
        assert(x2[i].shape[0]==sr.max_batch_seq_length)
        assert(np.any(x1[i]!=x2[i]))



def test_read_random_slice():
    

    sr = SequenceReader(h5_file, key_file, batch_size=5, max_seq_length=20,
                        seq_split_mode='random_slice', seq_split_overlap=5)

    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(len(x1)==int(sr.num_total_subseqs/sr.batch_size)*sr.batch_size)
    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]<=sr.max_batch_seq_length)
        assert(x1[i].shape[0]!=x2[i].shape[0] or
               np.all(x1[i]!=x2[i]))



def test_read_random_samples():
    

    sr = SequenceReader(h5_file, key_file, batch_size=5,
                        max_seq_length=20, min_seq_length=20,
                        seq_split_mode='random_samples', seq_split_overlap=5)

    #read epoch 1
    x1=[]
    for i in xrange(sr.num_batches):
        x1_i = sr.read()[0]
        assert(len(x1_i)==5)
        x1 += x1_i

    #read epoch 2
    x2=[]
    for i in xrange(sr.num_batches):
        x2_i = sr.read()[0]
        assert(len(x2_i)==5)
        x2 += x2_i

    assert(len(x1)==int(sr.num_total_subseqs/sr.batch_size)*sr.batch_size)
    assert(len(x1)==len(x2))
    for i in xrange(len(x1)):
        assert(x1[i].shape[0]==sr.max_batch_seq_length)
        assert(np.any(x1[i]!=x2[i]))

        
    
if __name__ == '__main__':
    pytest.main([__file__])




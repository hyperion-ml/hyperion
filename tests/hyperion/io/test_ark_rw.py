"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.hyp_defs import set_float_cpu, float_cpu
from hyperion.utils.kaldi_matrix import compression_methods

from hyperion.io.data_rw_factory import DataWriterFactory as DWF
from hyperion.io.data_rw_factory import SequentialDataReaderFactory as SDRF
from hyperion.io.data_rw_factory import RandomAccessDataReaderFactory as RDRF

input_prefix = './tests/data_in/ark/'
feat_scp_b = 'scp:./tests/data_in/ark/feat_b.scp'
feat_scp_t = 'scp:./tests/data_in/ark/feat_t.scp'
feat_scp_c = ['scp:./tests/data_in/ark/feat_c%d.scp' % i for i in xrange(1,8)]
feat_scp_uc = ['scp:./tests/data_in/ark/feat_uc%d.scp' % i for i in xrange(1,8)]
feat_ark_b = 'ark:./tests/data_in/ark/feat1_b.ark'
feat_ark_t = 'ark:./tests/data_in/ark/feat1_t.ark'
feat_ark_c = ['ark:./tests/data_in/ark/feat1_c%d.ark' % i for i in xrange(1,8)]
feat_ark_uc = ['ark:./tests/data_in/ark/feat1_uc%d.ark' % i for i in xrange(1,8)]

feat_range_b = 'scp:./tests/data_in/ark/feat_range_b.scp'
feat_range_c = ['scp:./tests/data_in/ark/feat_range_c%d.scp' % i for i in xrange(1,8)]

vec_scp_b = 'scp:./tests/data_in/ark/vec_b.scp'
vec_scp_t = 'scp:./tests/data_in/ark/vec_t.scp'
vec_ark_b = 'ark:./tests/data_in/ark/vec1_b.ark'
vec_ark_t = 'ark:./tests/data_in/ark/vec1_t.ark'

feat_ark_bo = 'ark:./tests/data_out/ark/feat.ark'
feat_scp_bo = 'scp:./tests/data_out/ark/feat.scp'
feat_ark_to = 'ark,t:./tests/data_out/ark/feat.ark'
feat_scp_co = ['scp:./tests/data_out/ark/feat_c%d.scp' % i for i in xrange(1,8)]
feat_both_bo = 'ark,scp:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'
feat_both_to = 'ark,scp,t:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'
feat_both_bfo = 'ark,scp,f:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'
feat_both_tfo = 'ark,scp,t,f:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'
feat_both_co = ['ark,scp:./tests/data_out/ark/feat_%d.ark,./tests/data_out/ark/feat_c%d.scp' % (i,i) for i in xrange(1,8)]

vec_ark_bo = 'ark:./tests/data_out/ark/feat.ark'
vec_scp_bo = 'scp:./tests/data_out/ark/feat.scp'
vec_ark_to = 'ark,t:./tests/data_out/ark/feat.ark'
vec_both_bo = 'ark,scp:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'
vec_both_to = 'ark,scp,t:./tests/data_out/ark/feat.ark,./tests/data_out/ark/feat.scp'



# Uncompressed feature files

def test_read_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0])

    # ark text
    r = SDRF.create(feat_ark_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


def test_read_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # scp text
    r = SDRF.create(feat_scp_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary
    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)

    # text
    r = RDRF.create(feat_scp_t, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2, rtol=1e-5)


def test_read_random_feat_permissive():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    key1.append('unk')
    
    # binary
    r = RDRF.create('p,'+feat_scp_b, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2)
    assert data2[-1].shape == (0,)

    # text
    r = RDRF.create('p,'+feat_scp_t, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2, rtol=1e-5)
    assert data2[-1].shape == (0,)



def test_read_seq_scp_split_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    key2 = []
    data2 = []
    for i in xrange(4):
        r = SDRF.create(feat_scp_b, path_prefix=input_prefix,
                        part_idx=i+1, num_parts=4)
        key_i, data_i = r.read(0)
        key2 = key2 + key_i
        data2 = data2 + data_i

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    
        
def test_write_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary
    w = DWF.create(feat_both_bo)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)


    # text
    w = DWF.create(feat_both_to)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        # i = np.isclose(d1,d2) == False
        # print(d1[i])
        # print(d2[i]) 

        assert_allclose(d1, d2, rtol=1e-4)



def test_write_flush_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary
    w = DWF.create(feat_both_bfo)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)


    # text
    w = DWF.create(feat_both_tfo)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        # i = np.isclose(d1,d2) == False
        # print(d1[i])
        # print(d2[i]) 

        assert_allclose(d1, d2, rtol=1e-4)



def test_with_read_seq_file_feat():

    # ark binary
    # without with
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0])

    # with with
    key2 = []
    data2 = []
    with SDRF.create(feat_ark_b, path_prefix=input_prefix) as r:
        while not r.eof():
            key_i, data_i = r.read(1)
            if len(key_i)==0:
                break
            key2.append(key_i[0])
            data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_with_read_seq_scp_feat():

    # scp binary
    # without with
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # with with
    key2 = []
    data2 = []
    with SDRF.create(feat_scp_t, path_prefix=input_prefix) as r:
        while not r.eof():
            key_i, data_i = r.read(1)
            key2.append(key_i[0])
            data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_with_read_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary with 
    with RDRF.create(feat_scp_b, path_prefix=input_prefix) as r:
        data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)
    
        

def test_with_write_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary with
    with DWF.create(feat_both_bo) as w:
        w.write(key1, data1)
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)



def test_read_iterator_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0])

        
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    for key_i, data_i in r:
        if len(key_i)==0:
            break
        key2.append(key_i)
        data2.append(data_i)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

        
    # ark text
    r = SDRF.create(feat_ark_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    for key_i, data_i in r:
        if len(key_i)==0:
            break
        key2.append(key_i)
        data2.append(data_i)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_read_iterator_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    for key_i, data_i in r:
        key2.append(key_i)
        data2.append(data_i)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

        
    # scp text
    r = SDRF.create(feat_scp_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    for key_i, data_i in r:
        key2.append(key_i)
        data2.append(data_i)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)



def test_reset_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0])

    # reset
    r.reset()
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_reset_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # reset
    r.reset()
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)


        
def test_read_shapes_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

        
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_shapes_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_shapes_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_num_rows_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])

    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_num_rows(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_num_rows_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])


    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_num_rows(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_num_rows_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])

    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    data2 = r.read_num_rows(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_dims_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])

        
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_dims(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_dims_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])


    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_dims(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_dims_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])

    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    data2 = r.read_dims(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_range_seq_scp_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50])
        i += 1

    r = SDRF.create(feat_range_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_range_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50])
        i += 1
        
    # binary
    r = RDRF.create(feat_range_b, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_range_shapes_seq_scp_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    r = SDRF.create(feat_range_b, path_prefix=input_prefix)
    key2, data2 = r.read_shapes(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_range_shapes_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1
        
    # binary
    r = RDRF.create(feat_range_b, path_prefix=input_prefix)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_range2_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1

    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1, row_offset=i, num_rows=10)
        if len(key_i)==0:
            break
        print(key_i[0])
        key2.append(key_i[0])
        data2.append(data_i[0])
        i += 1

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

        
        
def test_read_range2_seq_scp_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1, row_offset=i, num_rows=10)
        key2.append(key_i[0])
        data2.append(data_i[0])
        i += 1

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_range2_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    # binary
    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, row_offset=row_offset, num_rows=10)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_rangex2_seq_scp_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][2*i:2*i+10])
        i += 1

    r = SDRF.create(feat_range_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1, row_offset=i, num_rows=10)
        key2.append(key_i[0])
        data2.append(data_i[0])
        i += 1

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_rangex2_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][2*i:2*i+10])
        i += 1
        
    # binary
    r = RDRF.create(feat_range_b, path_prefix=input_prefix)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, row_offset=row_offset, num_rows=10)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_squeeze_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    # binary
    r = RDRF.create(feat_scp_b, path_prefix=input_prefix)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True, row_offset=row_offset, num_rows=10)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 3
    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


def test_read_squeeze_random_feat_permissive():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    # binary
    key1.append('unk')
    r = RDRF.create('p,'+feat_scp_b, path_prefix=input_prefix)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True, row_offset=row_offset, num_rows=10)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 3
    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2)
    assert_allclose(data2[-1], np.zeros(data2[0].shape))




def test_write_squeeze_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][:10])

    data1s = [np.expand_dims(d, axis=0) for d in data1]
    data1s = np.concatenate(tuple(data1s), axis=0)
    # binary
    w = DWF.create(feat_both_bo)
    w.write(key1, data1s)
    w.close()
    
    r = SDRF.create(feat_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)



# Compressed feature files

def test_read_compress_seq_file_feat():


    for i, cm in enumerate(compression_methods):
        # ark uncompressed binary
        r = SDRF.create(feat_ark_uc[i], path_prefix=input_prefix)
        key1, data1 = r.read(0)
        # key1 = []
        # data1 = []
        # while not r.eof():
        #     key_i, data_i = r.read(1)
        #     if len(key_i)==0:
        #         break
        #     key1.append(key_i[0])
        #     data1.append(data_i[0])

        # ark compressed
        r = SDRF.create(feat_ark_c[i], path_prefix=input_prefix)
        key2, data2 = r.read(0)
        # key2 = []
        # data2 = []
        # while not r.eof():
        #     key_i, data_i = r.read(1)
        #     if len(key_i)==0:
        #         break
        #     key2.append(key_i[0])
        #     data2.append(data_i[0])

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


def test_read_compress_seq_scp_feat():

    for i, cm in enumerate(compression_methods):
        # scp uncompressed binary
        r = SDRF.create(feat_scp_uc[i], path_prefix=input_prefix)
        key1, data1 = r.read(0)

        # scp compressed
        r = SDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        key2, data2 = r.read(0)

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


    
def test_read_compress_random_feat():

    for i, cm in enumerate(compression_methods):
        # scp uncompressed binary
        r = SDRF.create(feat_scp_uc[i], path_prefix=input_prefix)
        key1, data1 = r.read(0)

        # scp compressed
        r = RDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        data2 = r.read(key1)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_write_compress_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1, data1 = r.read(0)

    for i, cm in enumerate(compression_methods):
        # write compressed
        print('')
        w = DWF.create(feat_both_co[i], compress=True, compression_method=cm)
        w.write(key1, data1)
        w.close()

        # read compressed by kaldi copy-feats
        r = SDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        key1c, data1c = r.read(0)

        # read compressed
        r = SDRF.create(feat_scp_co[i])
        key2, data2 = r.read(0)

        for d1,d1c,d2 in zip(data1, data1c, data2):
            #idx = np.argmin(np.abs(d1))
            #atol = np.abs(d1.ravel()[idx]-d1c.ravel()[idx])
            #rtol = np.max(np.abs(np.abs(d1-d1c)-atol)/np.abs(d1))
            #f = np.isclose(d1, d2, rtol=rtol, atol=atol) == False
            err11c = np.abs(d1-d1c) + np.abs(d1)*0.001
            err1c2 = np.abs(d1c-d2)
            err12 = np.abs(d1-d2)
            
            f = np.logical_and(err11c < err1c2, err11c < err12)
            #print(atol, rtol)
            for a,b,c in zip(d1[f], d1c[f], d2[f]):
                print(a,b,c,a-b,b-c,a-c)
                
            assert not np.any(f), 'Write compression %s failed' % cm



def test_read_shapes_compress_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_ark_c[i], path_prefix=input_prefix)
        key2 = []
        data2 = []
        while not r.eof():
            key_i, data_i = r.read_shapes(1)
            if len(key_i)==0:
                break
            key2.append(key_i[0])
            data2.append(data_i[0])

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm


        
def test_read_shapes_compress_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        key2 = []
        data2 = []
        while not r.eof():
            key_i, data_i = r.read_shapes(1)
            key2.append(key_i[0])
            data2.append(data_i[0])

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm

    

def test_read_shapes_compress_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    for i, cm in enumerate(compression_methods):
        r = RDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        data2 = r.read_shapes(key1)

        for d1,d2 in zip(data1, data2):
            assert d1 == d2, 'Wrong shape for method %s' % cm



def test_read_range_compress_seq_scp_feat():

    for k, cm in enumerate(compression_methods):
        # scp uncompressed binary
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+50])
            i += 1

        # scp compressed
        print(feat_range_c[i])
        r = SDRF.create(feat_range_c[k], path_prefix=input_prefix)
        key2, data2 = r.read(0)
 
        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_range_compress_random_feat():

    for k, cm in enumerate(compression_methods):
        # scp uncompressed binary
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+50])
            i += 1
        
        # scp compressed binary
        r = RDRF.create(feat_range_c[k], path_prefix=input_prefix)
        data2 = r.read(key1)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))




def test_read_range_shapes_compress_seq_scp_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_range_c[k], path_prefix=input_prefix)
        key2, data2 = r.read_shapes(0)

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm

    

def test_read_range_shapes_compress_random_feat():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    for k, cm in enumerate(compression_methods):
        # compressed binary
        r = RDRF.create(feat_range_c[k], path_prefix=input_prefix)
        data2 = r.read_shapes(key1)
        for d1,d2 in zip(data1, data2):
            assert d1 == d2, 'Wrong shape for method %s' % cm



def test_read_range2_compress_seq_file_feat():

    for k, cm in enumerate(compression_methods):
        # ark uncompressed binary
        r = SDRF.create(feat_ark_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            if len(key_i)==0:
                break
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1
            
        # ark compressed binary
        r = SDRF.create(feat_ark_c[k], path_prefix=input_prefix)
        key2 = []
        data2 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1, row_offset=i, num_rows=10)
            if len(key_i)==0:
                break
            key2.append(key_i[0])
            data2.append(data_i[0])
            i += 1

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


        
def test_read_range2_compress_seq_scp_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1

        r = SDRF.create(feat_scp_c[k], path_prefix=input_prefix)
        key2 = []
        data2 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1, row_offset=i, num_rows=10)
            key2.append(key_i[0])
            data2.append(data_i[0])
            i += 1

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))

    

def test_read_range2_compress_random_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1
        
        r = RDRF.create(feat_scp_c[k], path_prefix=input_prefix)
        row_offset = [i for i in xrange(len(key1))]
        data2 = r.read(key1, row_offset=row_offset, num_rows=10)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_rangex2_compress_seq_scp_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][2*i:2*i+10])
            i += 1

        r = SDRF.create(feat_range_c[k], path_prefix=input_prefix)
        key2 = []
        data2 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1, row_offset=i, num_rows=10)
            key2.append(key_i[0])
            data2.append(data_i[0])
            i += 1

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))

    

def test_read_compress_rangex2_random_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_uc[k], path_prefix=input_prefix)
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][2*i:2*i+10])
            i += 1
        
        # binary
        r = RDRF.create(feat_range_c[k], path_prefix=input_prefix)
        row_offset = [i for i in xrange(len(key1))]
        data2 = r.read(key1, row_offset=row_offset, num_rows=10)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


            
# Vector files

def test_read_seq_file_vec():

    # ark binary
    r = SDRF.create(vec_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0])

    # ark text
    r = SDRF.create(vec_ark_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_read_seq_scp_vec():

    # scp binary
    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # scp text
    r = SDRF.create(vec_scp_t, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_random_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary
    r = RDRF.create(vec_scp_b, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)

    # text
    r = RDRF.create(vec_scp_t, path_prefix=input_prefix)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_write_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    # binary
    w = DWF.create(vec_both_bo)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(vec_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)


    # text
    w = DWF.create(vec_both_to)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(vec_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        i = np.isclose(d1,d2,rtol=1e-4, atol=1e-5) == False
        print(d1[i])
        print(d2[i]) 

        assert_allclose(d1, d2, rtol=1e-4, atol=1e-5)



def test_read_shapes_seq_file_vec():

    # ark binary
    r = SDRF.create(vec_ark_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

        
    r = SDRF.create(vec_ark_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        if len(key_i)==0:
            break
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2
        assert len(d1) == 1
        assert len(d2) == 1


        
def test_read_shapes_seq_scp_vec():

    # scp binary
    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2
        assert len(d1) == 1
        assert len(d2) == 1

    

def test_read_shapes_random_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(vec_scp_b, path_prefix=input_prefix)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2
        assert len(d1) == 1
        assert len(d2) == 1


        
def test_read_squeeze_random_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])
        i += 1
        
    # binary
    r = RDRF.create(vec_scp_b, path_prefix=input_prefix)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 2
    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_squeeze_random_vec_permissive():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])
        i += 1
        
    # binary
    key1.append('unk')
    r = RDRF.create('p,'+vec_scp_b, path_prefix=input_prefix)
    data2 = r.read(key1, squeeze=True)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 2
    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2)
    assert_allclose(data2[-1], np.zeros(data2[0].shape))




def test_write_squeeze_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    data1s = [np.expand_dims(d, axis=0) for d in data1]
    data1s = np.concatenate(tuple(data1s), axis=0)
    # binary
    w = DWF.create(vec_both_bo)
    w.write(key1, data1s)
    w.close()
    
    r = SDRF.create(vec_scp_bo)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)

# read compressed
# write compressed
# read compressed range x3
# read vector
# write vector

if __name__ == '__main__':
    pytest.main([__file__])

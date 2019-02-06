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
from hyperion.utils.list_utils import ismember
from hyperion.utils.kaldi_matrix import compression_methods

from hyperion.io.data_rw_factory import DataWriterFactory as DWF
from hyperion.io.data_rw_factory import SequentialDataReaderFactory as SDRF
from hyperion.io.data_rw_factory import RandomAccessDataReaderFactory as RDRF

input_prefix = './tests/data_in/ark/'
feat_scp_b = 'scp:./tests/data_in/ark/feat_b.scp'
feat_scp_c = ['scp:./tests/data_in/ark/feat_c%d.scp' % i for i in xrange(1,8)]
feat_ark_b = ['ark:./tests/data_in/ark/feat%d_b.ark' % i for i in xrange(1,3)]

vec_scp_b = 'scp:./tests/data_in/ark/vec_b.scp'

feat_h5_ho = ['h5:./tests/data_out/h5/feat%d.h5' %i for i in xrange(1,3)]
feat_scp_ho1 = ['./tests/data_out/h5/feat%d.scp' %i for i in xrange(1,3)]
feat_scp_ho2 = './tests/data_out/h5/feat.scp'
feat_scp_ho = 'scp:./tests/data_out/h5/feat.scp'
feat_range_ho1 = './tests/data_out/h5/feat_range.scp'
feat_range_ho = 'scp:./tests/data_out/h5/feat_range.scp'

feat_both_ho = ['h5,scp:./tests/data_out/h5/feat%d.h5,./tests/data_out/h5/feat%d.scp' %
                (i,i) for i in xrange(1,3)]
feat_scp_hso = 'scp:./tests/data_out/h5/feat_squeeze.scp'
feat_both_hso = 'h5,scp:./tests/data_out/h5/feat_squeeze.h5,./tests/data_out/h5/feat_squeeze.scp'


feat_scp_co = ['scp:./tests/data_out/ark/feat_c%d.scp' % i for i in xrange(1,8)]
feat_scp_hco = ['scp:./tests/data_out/h5/feat_c%d.scp' % i for i in xrange(1,8)]
feat_h5_hco = ['h5:./tests/data_out/h5/feat_c%d.h5' % i for i in xrange(1,8)]
feat_both_hco = ['h5,scp:./tests/data_out/h5/feat_c%d.h5,./tests/data_out/h5/feat_c%d.scp' % (i,i) for i in xrange(1,8)]

feat_scp_hco1 = ['./tests/data_out/h5/feat_c%d.scp' % i for i in xrange(1,8)]
feat_range_hco = ['scp:./tests/data_out/h5/feat_range_c%d.scp' % i for i in xrange(1,8)]
feat_range_hco1 = ['./tests/data_out/h5/feat_range_c%d.scp' % i for i in xrange(1,8)]

vec_h5_ho = 'h5:./tests/data_out/h5/vec.h5'
vec_scp_ho = 'scp:./tests/data_out/h5/vec.scp'
vec_both_ho = 'h5,scp:./tests/data_out/h5/vec.h5,./tests/data_out/h5/vec.scp'

vec_scp_hso = 'scp:./tests/data_out/h5/vec_squeeze.scp'
vec_both_hso = 'h5,scp:./tests/data_out/h5/vec_squeeze.h5,./tests/data_out/h5/vec_squeeze.scp'



# Uncompressed feature files

def test_write_read_seq_file_feat():

    for k in xrange(2):
        r = SDRF.create(feat_ark_b[k], path_prefix=input_prefix)
        key1, data1 = r.read(0)

        # write
        w = DWF.create(feat_both_ho[k])
        w.write(key1, data1)
        w.close()
    
        r = SDRF.create(feat_h5_ho[k])
        key2 = []
        data2 = []
        while not r.eof():
            key_i, data_i = r.read(1)
            key2.append(key_i[0])
            data2.append(data_i[0])


        f, loc = ismember(key1, key2)
        assert np.all(f)
        for i, (k1, d1) in enumerate(zip(key1,data1)):
            assert k1 == key2[loc[i]]
            assert_allclose(d1, data2[loc[i]])

    with open(feat_scp_ho2, 'w') as fw:
        for k in xrange(2):
            with open(feat_scp_ho1[k], 'r') as fr:
                for l in fr:
                    fw.write(l)



def test_write_flush_feat():


    r = SDRF.create(feat_ark_b[0], path_prefix=input_prefix)
    key1, data1 = r.read(0)
    
    # write
    w = DWF.create('f,'+feat_h5_ho[0])
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(feat_h5_ho[0])
    key2, data2 = r.read(0)

    f, loc = ismember(key1, key2)
    assert np.all(f)
    for i, (k1, d1) in enumerate(zip(key1,data1)):
        assert k1 == key2[loc[i]]
        assert_allclose(d1, data2[loc[i]])



def test_with_write_feat():

    r = SDRF.create(feat_ark_b[0], path_prefix=input_prefix)
    key1, data1 = r.read(0)
    
    # write
    with DWF.create(feat_h5_ho[0]) as w:
        w.write(key1, data1)
    
    r = SDRF.create(feat_h5_ho[0])
    key2, data2 = r.read(0)

    f, loc = ismember(key1, key2)
    assert np.all(f)
    for i, (k1, d1) in enumerate(zip(key1,data1)):
        assert k1 == key2[loc[i]]
        assert_allclose(d1, data2[loc[i]])



def test_read_seq_scp_feat():

    # ark binary
    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1, data1 = r.read(0)

    # h5
    r = SDRF.create(feat_scp_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)



def test_read_random_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)

    r = RDRF.create(feat_h5_ho[0])
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)

    r = RDRF.create(feat_scp_ho)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_random_file_feat_permissive():

    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)
    key1.append('unk')
    data1.append(np.array([]))

    r = RDRF.create('p,'+feat_h5_ho[0])
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_random_scp_feat_permissive():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)
    key1.append('unk')
    data1.append(np.array([]))

    r = RDRF.create('p,'+feat_scp_ho)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_seq_file_split_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)

    key2 = []
    data2 = []
    for i in xrange(2):
        r = SDRF.create(feat_h5_ho[0], part_idx=i+1, num_parts=2)
        key_i, data_i = r.read(0)
        key2 += key_i
        data2 += data_i
    
    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)



def test_read_seq_scp_split_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)

    key2 = []
    data2 = []
    for i in xrange(4):
        r = SDRF.create(feat_scp_ho, part_idx=i+1, num_parts=4)
        key_i, data_i = r.read(0)
        key2 += key_i
        data2 += data_i
    
    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)


        
def test_with_read_seq_file_feat():

    # without with
    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)
 
    # with with
    with SDRF.create(feat_h5_ho[0]) as r:
        key2, data2 = r.read(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)


        
def test_with_read_seq_scp_feat():

    # without with
    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)
 
    # with with
    with SDRF.create(feat_scp_ho) as r:
        key2, data2 = r.read(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

    
    

def test_with_read_random_file_feat():

    # without with
    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)
 
    # with with
    with RDRF.create(feat_h5_ho[0]) as r:
        data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_with_read_random_scp_feat():

    # without with
    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)
 
    # with with
    with RDRF.create(feat_scp_ho) as r:
        data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)

        

def test_read_iterator_seq_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)
        
    r = SDRF.create(feat_h5_ho[0])
    key2 = []
    data2 = []
    for key_i, data_i in r:
        key2.append(key_i)
        data2.append(data_i)
    print(key1)
    print(key2)
    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

        
        
def test_read_iterator_seq_scp_feat():

    # scp binary
    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)

    r = SDRF.create(feat_scp_ho)
    key2 = []
    data2 = []
    for key_i, data_i in r:
        key2.append(key_i)
        data2.append(data_i)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-5)

        

def test_reset_seq_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1, data1 = r.read(0)

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
    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)

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

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

        
    r = SDRF.create(feat_h5_ho[0])
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
    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    r = SDRF.create(feat_scp_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_shapes_random_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(feat_h5_ho[0])
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_shapes_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(feat_scp_ho)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_num_rows_seq_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])

    r = SDRF.create(feat_h5_ho[0])
    key2, data2 = r.read_num_rows(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_num_rows_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])


    r = SDRF.create(feat_scp_ho)
    key2, data2 = r.read_num_rows(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_num_rows_random_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])

    r = RDRF.create(feat_h5_ho[0])
    data2 = r.read_num_rows(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_num_rows_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[0])

    r = RDRF.create(feat_scp_ho)
    data2 = r.read_num_rows(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_dims_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])

        
    r = SDRF.create(feat_h5_ho[0])
    key2, data2 = r.read_dims(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_dims_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])


    r = SDRF.create(feat_scp_ho)
    key2, data2 = r.read_dims(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_dims_random_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])

    r = RDRF.create(feat_h5_ho[0])
    data2 = r.read_dims(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_dims_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape[1])

    r = RDRF.create(feat_scp_ho)
    data2 = r.read_dims(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_range_seq_scp_feat():

    with open(feat_range_ho1, 'w') as w:
        with open(feat_scp_ho2, 'r') as r:
            i = 0
            for l in r:
                w.write('%s[%d:%d]\n' % (l.strip(), i, i+50))
                i += 1
    
    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50])
        i += 1

    r = SDRF.create(feat_range_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)

    

def test_read_range_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50])
        i += 1
        
    # binary
    r = RDRF.create(feat_range_ho)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_range_shapes_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    r = SDRF.create(feat_range_ho)
    key2, data2 = r.read_shapes(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2

    

def test_read_range_shapes_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1
        
    r = RDRF.create(feat_range_ho)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_range2_seq_file_feat():

    # ark binary
    r = SDRF.create(feat_h5_ho[0])
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

    r = SDRF.create(feat_h5_ho[0])
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

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1

    r = SDRF.create(feat_scp_ho)
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

    

def test_read_range2_random_file_feat():

    r = SDRF.create(feat_h5_ho[0])
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    r = RDRF.create(feat_h5_ho[0])
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, row_offset=row_offset, num_rows=10)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_range2_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    # binary
    r = RDRF.create(feat_scp_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, row_offset=row_offset, num_rows=10)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_rangex2_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][2*i:2*i+10])
        i += 1

    r = SDRF.create(feat_range_ho)
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

    

def test_read_rangex2_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][2*i:2*i+10])
        i += 1
        
    # binary
    r = RDRF.create(feat_range_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, row_offset=row_offset, num_rows=10)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_squeeze_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    r = RDRF.create(feat_scp_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True, row_offset=row_offset, num_rows=10)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 3
    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_squeeze_random_scp_feat_permissive():

    r = SDRF.create(feat_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+10])
        i += 1
        
    key1.append('unk')
    r = RDRF.create('p,'+feat_scp_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True, row_offset=row_offset, num_rows=10)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 3
    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2)
    assert_allclose(data2[-1], np.zeros(data2[0].shape))




def test_write_squeeze_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][:10])

    data1s = [np.expand_dims(d, axis=0) for d in data1]
    data1s = np.concatenate(tuple(data1s), axis=0)

    w = DWF.create(feat_both_hso)
    w.write(key1, data1s)
    w.close()
    
    r = SDRF.create(feat_scp_hso)
    key2, data2 = r.read(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)



# Compressed feature files

def test_write_read_seq_scp_compress_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read(0)

    for i, cm in enumerate(compression_methods):
        # write compressed
        print('')
        w = DWF.create(feat_both_hco[i], compress=True, compression_method=cm)
        w.write(key1, data1)
        w.close()

        # read kaldi compressed
        r = SDRF.create(feat_scp_c[i], path_prefix=input_prefix)
        key1c, data1c = r.read(0)

        # read compressed
        r = SDRF.create(feat_scp_hco[i])
        key2, data2 = r.read(0)

        for d1,d1c,d2 in zip(data1, data1c, data2):
            err11c = np.abs(d1-d1c) + np.abs(d1)*0.001
            err1c2 = np.abs(d1c-d2)
            err12 = np.abs(d1-d2)
            
            f = np.logical_and(err11c < err1c2, err11c < err12)
            for a,b,c in zip(d1[f], d1c[f], d2[f]):
                print(a,b,c,a-b,b-c,a-c)
                
            assert not np.any(f), 'Write compression %s failed' % cm



def test_read_compress_seq_file_feat():

    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_h5_hco[i])
        key1, data1 = r.read(0)

        r = SDRF.create(feat_scp_hco[i])
        key2, data2 = r.read(0)

        f, loc = ismember(key2, key1)
        for i,(k2,d2) in enumerate(zip(key2, data2)):
            assert key1[loc[i]] == k2
            assert_allclose(data1[loc[i]], d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


    
def test_read_compress_random_file_feat():

    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[i])
        key1, data1 = r.read(0)

        r = RDRF.create(feat_h5_hco[i])
        data2 = r.read(key1)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_compress_random_scp_feat():

    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[i])
        key1, data1 = r.read(0)

        r = RDRF.create(feat_scp_hco[i])
        data2 = r.read(key1)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))


    
def test_read_shapes_compress_seq_file_feat():

    r = SDRF.create(feat_h5_hco[0])
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_h5_hco[i])
        key2, data2 = r.read_shapes(0)

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm


        
def test_read_shapes_compress_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read_shapes(0)

    for i, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[i])
        key2, data2 = r.read_shapes(0)

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm

    

def test_read_shapes_compress_random_file_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read_shapes(0)

    for i, cm in enumerate(compression_methods):
        r = RDRF.create(feat_h5_hco[i])
        data2 = r.read_shapes(key1)

        for d1,d2 in zip(data1, data2):
            assert d1 == d2, 'Wrong shape for method %s' % cm


            
def test_read_shapes_compress_random_file_feat():

    r = SDRF.create(feat_scp_ho)
    key1, data1 = r.read_shapes(0)

    for i, cm in enumerate(compression_methods):
        r = RDRF.create(feat_scp_hco[i])
        data2 = r.read_shapes(key1)

        for d1,d2 in zip(data1, data2):
            assert d1 == d2, 'Wrong shape for method %s' % cm



def test_read_range_compress_seq_scp_feat():

    
    for k, cm in enumerate(compression_methods):
        with open(feat_range_hco1[k], 'w') as w:
            with open(feat_scp_hco1[k], 'r') as r:
                i = 0
                for l in r:
                    w.write('%s[%d:%d]\n' % (l.strip(), i, i+50))
                    i += 1

        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+50])
            i += 1

        r = SDRF.create(feat_range_hco[k])
        key2, data2 = r.read(0)
 
        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_range_compress_random_feat():

    for k, cm in enumerate(compression_methods):
        # scp uncompressed binary
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+50])
            i += 1
        
        # scp compressed binary
        r = RDRF.create(feat_range_hco[k])
        data2 = r.read(key1)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_range_shapes_compress_seq_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_range_hco[k])
        key2, data2 = r.read_shapes(0)

        for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
            assert k1 == k2, 'Wrong key for method %s' % cm
            assert d1 == d2, 'Wrong shape for method %s' % cm

    

def test_read_range_shapes_compress_random_scp_feat():

    r = SDRF.create(feat_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0][i:i+50].shape)
        i += 1

    for k, cm in enumerate(compression_methods):
        r = RDRF.create(feat_range_hco[k])
        data2 = r.read_shapes(key1)
        for d1,d2 in zip(data1, data2):
            assert d1 == d2, 'Wrong shape for method %s' % cm



def test_read_range2_compress_seq_file_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_h5_hco[k])
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
            
        r = SDRF.create(feat_h5_hco[k])
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
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1

        r = SDRF.create(feat_scp_hco[k])
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

    

def test_read_range2_compress_random_file_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1
        
        r = RDRF.create(feat_h5_hco[k])
        row_offset = [i for i in xrange(len(key1))]
        data2 = r.read(key1, row_offset=row_offset, num_rows=10)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_range2_compress_random_file_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][i:i+10])
            i += 1
        
        r = RDRF.create(feat_scp_hco[k])
        row_offset = [i for i in xrange(len(key1))]
        data2 = r.read(key1, row_offset=row_offset, num_rows=10)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



def test_read_rangex2_compress_seq_scp_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][2*i:2*i+10])
            i += 1

        r = SDRF.create(feat_range_hco[k])
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

    

def test_read_compress_rangex2_random_file_feat():

    for k, cm in enumerate(compression_methods):
        r = SDRF.create(feat_scp_hco[k])
        key1 = []
        data1 = []
        i = 0
        while not r.eof():
            key_i, data_i = r.read(1)
            key1.append(key_i[0])
            data1.append(data_i[0][2*i:2*i+10])
            i += 1
        
        r = RDRF.create(feat_range_hco[k])
        row_offset = [i for i in xrange(len(key1))]
        data2 = r.read(key1, row_offset=row_offset, num_rows=10)

        for d1,d2 in zip(data1, data2):
            assert_allclose(d1, d2, rtol=1e-5, atol=1e-4,
                            err_msg=('Read compression %s failed' % cm))



# Vector files

def test_write_read_seq_file_vec():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1, data1 = r.read(0)
    
    # write
    w = DWF.create(vec_both_ho)
    w.write(key1, data1)
    w.close()
    
    r = SDRF.create(vec_h5_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    f, loc = ismember(key1, key2)
    assert np.all(f)
    for i, (k1, d1) in enumerate(zip(key1,data1)):
        assert k1 == key2[loc[i]]
        assert_allclose(d1, data2[loc[i]])



def test_read_seq_scp_vec():

    # ark binary
    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1, data1 = r.read(0)

    # h5
    r = SDRF.create(vec_scp_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2, rtol=1e-4)



def test_read_random_file_vec():

    r = SDRF.create(vec_h5_ho)
    key1, data1 = r.read(0)

    r = RDRF.create(vec_h5_ho)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_random_scp_vec():

    r = SDRF.create(vec_scp_ho)
    key1, data1 = r.read(0)

    r = RDRF.create(vec_scp_ho)
    data2 = r.read(key1)

    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)



def test_read_shapes_seq_file_vec():

    r = SDRF.create(vec_h5_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        if len(key_i)==0:
            break
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

        
    r = SDRF.create(vec_h5_ho)
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


        
def test_read_shapes_seq_scp_vec():

    r = SDRF.create(vec_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)


    r = SDRF.create(vec_scp_ho)
    key2 = []
    data2 = []
    while not r.eof():
        key_i, data_i = r.read_shapes(1)
        key2.append(key_i[0])
        data2.append(data_i[0])

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert d1 == d2


        
def test_read_shapes_random_file_vec():

    r = SDRF.create(vec_h5_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(vec_h5_ho)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2


        
def test_read_shapes_random_scp_vec():

    r = SDRF.create(vec_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0].shape)

    r = RDRF.create(vec_scp_ho)
    data2 = r.read_shapes(key1)

    for d1,d2 in zip(data1, data2):
        assert d1 == d2



def test_read_squeeze_random_scp_vec():

    r = SDRF.create(vec_scp_ho)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])
        i += 1
        
    r = RDRF.create(vec_scp_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 2
    for d1,d2 in zip(data1, data2):
        assert_allclose(d1, d2)


        
def test_read_squeeze_random_scp_vec_permissive():

    r = SDRF.create(vec_scp_b, path_prefix=input_prefix)
    key1 = []
    data1 = []
    i = 0
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])
        i += 1
        
    key1.append('unk')
    r = RDRF.create('p,'+vec_scp_ho)
    row_offset = [i for i in xrange(len(key1))]
    data2 = r.read(key1, squeeze=True)

    assert isinstance(data2, np.ndarray)
    assert data2.ndim == 2
    for d1,d2 in zip(data1, data2[:-1]):
        assert_allclose(d1, d2)
    assert_allclose(data2[-1], np.zeros(data2[0].shape))



def test_write_squeeze_vec():

    r = SDRF.create(vec_scp_ho)
    key1 = []
    data1 = []
    while not r.eof():
        key_i, data_i = r.read(1)
        key1.append(key_i[0])
        data1.append(data_i[0])

    data1s = [np.expand_dims(d, axis=0) for d in data1]
    data1s = np.concatenate(tuple(data1s), axis=0)

    w = DWF.create(vec_both_hso)
    w.write(key1, data1s)
    w.close()
    
    r = SDRF.create(vec_scp_hso)
    key2, data2 = r.read(0)

    for k1,k2,d1,d2 in zip(key1, key2, data1, data2):
        assert k1 == k2
        assert_allclose(d1, d2)


if __name__ == '__main__':
    pytest.main([__file__])

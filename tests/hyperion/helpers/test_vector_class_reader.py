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
import numpy as np
from numpy.testing import assert_allclose

from hyperion.utils.utt2info import Utt2Info
from hyperion.io import H5DataWriter
from hyperion.helpers import VectorClassReader

output_dir = './tests/data_out/helpers'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_u2c():
    key = [str(k) for k in xrange(10)]
    classes = [ 'c1' ] + ['c3']*6 + [ 'c2' ]*3 
    u2c = Utt2Info.create(key, classes)
    return u2c


def test__filter_by_spc_min_spc():
    u2c_in = create_u2c()
    
    u2c_out = VectorClassReader._filter_by_spc(u2c_in, min_spc=2)
    u2c_gt = Utt2Info.create(u2c_in.key[1:], u2c_in.info[1:])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, min_spc=4)
    u2c_gt = Utt2Info.create(u2c_in.key[1:7], u2c_in.info[1:7])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, min_spc=7)
    u2c_gt = Utt2Info.create([], [])
    assert u2c_out==u2c_gt

    
def test__filter_by_spc_max_spc():
    
    u2c_in = create_u2c()

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, max_spc=4,
                                             spc_pruning_mode='last')
    f = np.ones_like(u2c_in.key, dtype=bool)
    f[5:7] = False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, max_spc=4,
                                             spc_pruning_mode='first')
    f = np.ones_like(u2c_in.key, dtype=bool)
    f[1:3] = False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)    
    u2c_out = VectorClassReader._filter_by_spc(u2c_in, max_spc=4,
                                               spc_pruning_mode='random', rng=rng)

    f = np.ones_like(u2c_in.key, dtype=bool)
    f[3] = False
    f[6] = False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt


def test__filter_by_spc_min_max_spc():
    
    u2c_in = create_u2c()

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='last')
    f = np.ones_like(u2c_in.key, dtype=bool)
    f[0] = False
    f[5:7] = False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._filter_by_spc(u2c_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='first')
    f = np.ones_like(u2c_in.key, dtype=bool)
    f[:3]=False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out=VectorClassReader._filter_by_spc(u2c_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='random', rng=rng)
    f = np.ones_like(u2c_in.key, dtype=bool)
    f[0] = False
    f[3] = False
    f[6] = False
    u2c_gt = Utt2Info.create(u2c_in.key[f], u2c_in.info[f])
    assert u2c_out==u2c_gt


def test__split_classes_sequential_nonoverlap():

    u2c_in = create_u2c()

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=3, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','4','5','6'],
        ['0','1','1','1','2','2','2','3','3','3'])
    assert u2c_out==u2c_gt
    
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=2, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','4','5','6'],
        ['0','1','1','2','3','3','4','4','5','5'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=4, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','4','5','6'],
        ['0','1','1','1','2','2','2','2','3','3'])
    assert u2c_out==u2c_gt
 
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=2, max_spc=3, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['7','8','9','1','2','3','4','5','6'],
        ['1','1','1','2','2','2','3','3','3'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=2, max_spc=2, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['7','8','1','2','3','4','5','6'],
        ['1','1','3','3','4','4','5','5'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=4, max_spc=4, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['1','2','3','4'], ['2','2','2','2'])
    assert u2c_out==u2c_gt


def test__split_classes_random_nonoverlap():
    
    u2c_in = create_u2c()

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=3, mode='random', rng=rng)
    print(u2c_out.key)
    print(u2c_out.info)
    u2c_gt = Utt2Info.create(['0','1','1','1','2','2','2','3','3','3'],
                     ['0','7','8','9','3','6','1','3','4','5'])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=2, mode='random', rng=rng)
    print(u2c_out.key)
    print(u2c_out.info)
    u2c_gt = Utt2Info.create(['0','1','1','2','2','3','3','4','4','5','5'],
                     ['0','7','9','9','8','3','4','5','2','3','1'])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=4, mode='random', rng=rng)
    print(u2c_out.key)
    print(u2c_out.info)
    u2c_gt = Utt2Info.create(['0','1','1','1','2','2','2','2','3','3','3','3'],
                     ['0','7','8','9','3','6','1','5','3','4','5','1'])
    assert u2c_out==u2c_gt


    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=2, max_spc=3, mode='random', rng=rng)
    print(u2c_out.key)
    print(u2c_out.info)
    u2c_gt = Utt2Info.create(['1','1','1','2','2','2','3','3','3'],
                     ['7','8','9','3','6','1','3','4','5'])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=2, max_spc=2, mode='random', rng=rng)
    u2c_gt = Utt2Info.create(['1','1','2','2','3','3','4','4','5','5'],
                     ['7','9','9','8','3','4','5','2','3','1'])
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=4, max_spc=4, mode='random', rng=rng)
    u2c_gt = Utt2Info.create(['2','2','2','2','3','3','3','3'],
                     ['3','6','1','5','3','4','5','1'])
    assert u2c_out==u2c_gt



def test__split_classes_sequential_overlap():

    u2c_in = create_u2c()

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=3, overlap=1, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','3','4','5','5','6'],
        ['0','1','1','1','2','2','2','3','3','3','4','4'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=3, overlap=2, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','2','3','4','3','4','5','4','5','6'],
        ['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=2, overlap=1, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','8','9','1','2','2','3','3','4','4','5','5','6'],
        ['0','1','1','2','2','3','3','4','4','5','5','6','6','7','7'])
    assert u2c_out==u2c_gt

    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=4, overlap=3, mode='sequential')
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','1','2','3','4','2','3','4','5','3','4','5','6'],
        ['0','1','1','1','2','2','2','2','3','3','3','3','4','4','4','4'])
    assert u2c_out==u2c_gt


    
def test__split_classes_random_nonoverlap():
    
    u2c_in = create_u2c()

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=3, overlap=2, mode='random', rng=rng)
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','3','6','1','3','4','5','5','2','6','3','1','5'],
        ['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'])

    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=2, overlap=1, mode='random', rng=rng)
    u2c_gt = Utt2Info.create(
        ['0','7','9','9','8','3','4','5','2','3','1','3','5','3','2'],
        ['0','1','1','2','2','3','3','4','4','5','5','6','6','7','7'])
                     
    assert u2c_out==u2c_gt

    rng = np.random.RandomState(1024)
    u2c_out = VectorClassReader._split_classes(
        u2c_in, min_spc=1, max_spc=4, overlap=2, mode='random', rng=rng)
    u2c_gt = Utt2Info.create(
        ['0','7','8','9','3','6','1','5','3','4','5','1'],
        ['0','1','1','1','2','2','2','2','3','3','3','3'])

    assert u2c_out==u2c_gt


    
def test_vector_class_reader():

    v_file = output_dir + '/vcr.h5'
    key_file = output_dir + '/vcr.u2c'

    u2c = create_u2c()
    x = np.random.randn(len(u2c.key),2).astype('float32')
    
    h = H5DataWriter(v_file)
    h.write(u2c.key, x)
    u2c.save(key_file)

    vcr = VectorClassReader(v_file, key_file, vlist_sep=' ',
                            csplit_min_spc=1, csplit_max_spc=3,
                            csplit_overlap=2, csplit_mode='random',
                            vcr_seed=1024)
    
    x_test, class_ids_test = vcr.read()
    print(x_test)
    print(class_ids_test)

    u2c_gt = Utt2Info.create(['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'],
                             ['0','7','8','9','3','6','1','3','4','5','5','2','6','3','1','5'])
    class_ids_gt = np.array([0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5], dtype='int')
    indx_gt = np.array([0,7,8,9,3,6,1,3,4,5,5,2,6,3,1,5], dtype='int')
    x_gt = x[indx_gt, :]
    print(x_gt)
    print(class_ids_gt)
    assert_allclose(x_test, x_gt, rtol=1e-5)
    assert np.all(class_ids_test==class_ids_gt)

    
if __name__ == '__main__':
    pytest.main([__file__])

    
    


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hyperion.utils.scp_list import SCPList
from hyperion.io import HypDataWriter
from hyperion.helpers import VectorClassReader


def create_scp():
    file_path = [str(k) for k in xrange(10)]
    key = [ 'c1' ] + ['c3']*6 + [ 'c2' ]*3 
    scp = SCPList(key, file_path)
    return scp


def test__filter_by_spc_min_scp():
    scp_in = create_scp()
    
    scp_out = VectorClassReader._filter_by_spc(scp_in, min_spc=2)
    scp_gt = SCPList(scp_in.key[1:], scp_in.file_path[1:])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._filter_by_spc(scp_in, min_spc=4)
    scp_gt = SCPList(scp_in.key[1:7], scp_in.file_path[1:7])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._filter_by_spc(scp_in, min_spc=7)
    scp_gt = SCPList([], [])
    assert(scp_out==scp_gt)

    
def test__filter_by_spc_max_scp():
    
    scp_in = create_scp()

    scp_out = VectorClassReader._filter_by_spc(scp_in, max_spc=4,
                                             spc_pruning_mode='last')
    f = np.ones_like(scp_in.key, dtype=bool)
    f[5:7] = False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._filter_by_spc(scp_in, max_spc=4,
                                             spc_pruning_mode='first')
    f = np.ones_like(scp_in.key, dtype=bool)
    f[1:3] = False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)    
    scp_out = VectorClassReader._filter_by_spc(scp_in, max_spc=4,
                                               spc_pruning_mode='random', rng=rng)

    f = np.ones_like(scp_in.key, dtype=bool)
    f[3] = False
    f[6] = False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)


def test__filter_by_spc_min_max_scp():
    
    scp_in = create_scp()

    scp_out = VectorClassReader._filter_by_spc(scp_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='last')
    f = np.ones_like(scp_in.key, dtype=bool)
    f[0] = False
    f[5:7] = False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._filter_by_spc(scp_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='first')
    f = np.ones_like(scp_in.key, dtype=bool)
    f[:3]=False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out=VectorClassReader._filter_by_spc(scp_in, min_spc=2, max_spc=4,
                                             spc_pruning_mode='random', rng=rng)
    f = np.ones_like(scp_in.key, dtype=bool)
    f[0] = False
    f[3] = False
    f[6] = False
    scp_gt = SCPList(scp_in.key[f], scp_in.file_path[f])
    assert(scp_out==scp_gt)


def test__split_classes_sequential_nonoverlap():

    scp_in = create_scp()

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=3, mode='sequential')
    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3'],
                     ['0','7','8','9','1','2','3','4','5','6'])
    assert(scp_out==scp_gt)
    
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=2, mode='sequential')
    scp_gt = SCPList(['0','1','1','2','3','3','4','4','5','5'],
                     ['0','7','8','9','1','2','3','4','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=4, mode='sequential')
    scp_gt = SCPList(['0','1','1','1','2','2','2','2','3','3'],
                     ['0','7','8','9','1','2','3','4','5','6'])
    assert(scp_out==scp_gt)
 
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=2, max_spc=3, mode='sequential')
    scp_gt = SCPList(['1','1','1','2','2','2','3','3','3'],
                     ['7','8','9','1','2','3','4','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=2, max_spc=2, mode='sequential')
    scp_gt = SCPList(['1','1','3','3','4','4','5','5'],
                     ['7','8','1','2','3','4','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=4, max_spc=4, mode='sequential')
    scp_gt = SCPList(['2','2','2','2'],
                     ['1','2','3','4'])
    assert(scp_out==scp_gt)


def test__split_classes_random_nonoverlap():
    
    scp_in = create_scp()

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=3, mode='random', rng=rng)
    print(scp_out.key)
    print(scp_out.file_path)
    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3'],
                     ['0','7','8','9','3','6','1','3','4','5'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=2, mode='random', rng=rng)
    print(scp_out.key)
    print(scp_out.file_path)
    scp_gt = SCPList(['0','1','1','2','2','3','3','4','4','5','5'],
                     ['0','7','9','9','8','3','4','5','2','3','1'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=4, mode='random', rng=rng)
    print(scp_out.key)
    print(scp_out.file_path)
    scp_gt = SCPList(['0','1','1','1','2','2','2','2','3','3','3','3'],
                     ['0','7','8','9','3','6','1','5','3','4','5','1'])
    assert(scp_out==scp_gt)


    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=2, max_spc=3, mode='random', rng=rng)
    print(scp_out.key)
    print(scp_out.file_path)
    scp_gt = SCPList(['1','1','1','2','2','2','3','3','3'],
                     ['7','8','9','3','6','1','3','4','5'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=2, max_spc=2, mode='random', rng=rng)
    scp_gt = SCPList(['1','1','2','2','3','3','4','4','5','5'],
                     ['7','9','9','8','3','4','5','2','3','1'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=4, max_spc=4, mode='random', rng=rng)
    scp_gt = SCPList(['2','2','2','2','3','3','3','3'],
                     ['3','6','1','5','3','4','5','1'])
    assert(scp_out==scp_gt)



def test__split_classes_sequential_overlap():

    scp_in = create_scp()

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=3, overlap=1, mode='sequential')
    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3','4','4'],
                     ['0','7','8','9','1','2','3','3','4','5','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=3, overlap=2, mode='sequential')
    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'],
                     ['0','7','8','9','1','2','3','2','3','4','3','4','5','4','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=2, overlap=1, mode='sequential')
    scp_gt = SCPList(['0','1','1','2','2','3','3','4','4','5','5','6','6','7','7'],
                     ['0','7','8','8','9','1','2','2','3','3','4','4','5','5','6'])
    assert(scp_out==scp_gt)

    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=4, overlap=3, mode='sequential')
    scp_gt = SCPList(['0','1','1','1','2','2','2','2','3','3','3','3','4','4','4','4'],
                     ['0','7','8','9','1','2','3','4','2','3','4','5','3','4','5','6'])
    assert(scp_out==scp_gt)


    
def test__split_classes_random_nonoverlap():
    
    scp_in = create_scp()

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=3, overlap=2, mode='random', rng=rng)
    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'],
                     ['0','7','8','9','3','6','1','3','4','5','5','2','6','3','1','5'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=2, overlap=1, mode='random', rng=rng)
    scp_gt = SCPList(['0','1','1','2','2','3','3','4','4','5','5','6','6','7','7'],
                     ['0','7','9','9','8','3','4','5','2','3','1','3','5','3','2'])
    assert(scp_out==scp_gt)

    rng = np.random.RandomState(1024)
    scp_out = VectorClassReader._split_classes(
        scp_in, min_spc=1, max_spc=4, overlap=2, mode='random', rng=rng)
    scp_gt = SCPList(['0','1','1','1','2','2','2','2','3','3','3','3'],
                     ['0','7','8','9','3','6','1','5','3','4','5','1'])
    assert(scp_out==scp_gt)

    
def test_vector_class_reader():

    v_file = './tests/data_out/vcr.h5'
    key_file = './tests/data_out/vcr.scp'

    scp = create_scp()
    x = np.random.randn(len(scp.key),2).astype('float32')
    
    h = HypDataWriter(v_file)
    h.write(scp.file_path, '', x)
    scp.save(key_file)

    vcr = VectorClassReader(v_file, key_file, scp_sep=' ',
                            csplit_min_spc=1, csplit_max_spc=3,
                            csplit_overlap=2, csplit_mode='random', seed=1024)
    
    x_test, class_ids_test = vcr.read()
    print(x_test)
    print(class_ids_test)

    scp_gt = SCPList(['0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'],
                     ['0','7','8','9','3','6','1','3','4','5','5','2','6','3','1','5'])
    class_ids_gt = np.array([0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5], dtype='int')
    indx_gt = np.array([0,7,8,9,3,6,1,3,4,5,5,2,6,3,1,5], dtype='int')
    x_gt = x[indx_gt, :]
    print(x_gt)
    print(class_ids_gt)
    assert_allclose(x_test, x_gt, rtol=1e-5)
    assert(np.all(class_ids_test==class_ids_gt))

    
if __name__ == '__main__':
    pytest.main([__file__])

    
    

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

from hyperion.utils.kaldi_matrix import KaldiMatrix as KM
from hyperion.utils.kaldi_matrix import KaldiCompressedMatrix as KCM

output_dir = './tests/data_out/utils/kaldi_matrix'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_matrix(r=10, c=4):
    return np.random.randn(r, c)

def create_matrix_int(r=10, c=4):
    return 32000*(np.random.rand(r, c)-0.5)

def create_matrix_uint(r=10, c=4):
    return 250*np.random.rand(r, c)

def create_matrix_01(r=10, c=4):
    return np.random.rand(r, c)

def create_vector(d=10):
    return np.random.randn(d)


def test_km_read_write():

    file_path = output_dir + '/km.mat'

    # Test Matrix
    mat1 = KM(create_matrix().astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=False)

    assert_allclose(mat1.data, mat2.data, atol=1e-4)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True)

    assert_allclose(mat1.data, mat2.data)

    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True, row_offset=5)

    assert_allclose(mat1.data[5:], mat2.data)

    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True, row_offset=4, num_rows=4)

    assert_allclose(mat1.data[4:8], mat2.data)


    mat1 = KM(mat1.data.astype('float64'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=False)

    assert_allclose(mat1.data, mat2.data, atol=1e-4)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True)

    assert_allclose(mat1.data, mat2.data)

    # Test Vector
    mat1 = KM(create_vector().astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=False)

    assert_allclose(mat1.data, mat2.data, atol=1e-4)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True)

    assert_allclose(mat1.data, mat2.data)


    mat1 = KM(mat1.data.astype('float64'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=False)

    assert_allclose(mat1.data, mat2.data, atol=1e-4)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        mat2 = KM.read(f, binary=True)

    assert_allclose(mat1.data, mat2.data)



def test_km_read_shape():

    file_path = output_dir + '/km.mat'

    # Test Matrix
    mat1 = KM(create_matrix(10,4).astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=False) == (10, 4)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=True) == (10, 4)


    mat1 = KM(mat1.data.astype('float64'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=False) == (10, 4)

        
    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=True) == (10, 4)


    # Test Vector
    mat1 = KM(create_vector(10).astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=False) == (10,)

        
    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=True) == (10,)


    mat1 = KM(mat1.data.astype('float64'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=False) == (10,)

        
    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        assert KM.read_shape(f, binary=True) == (10,)


    
def test_kcm_compress():

    mat1 = KM(create_matrix().astype('float32'))
    cmat2 = KCM.compress(mat1, 'auto')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=0.025)

    cmat2 = KCM.compress(mat1, '2byte-auto')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=0.025)

    cmat2 = KCM.compress(mat1, '1byte-auto')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=0.025)

    
    mat1 = KM(create_matrix(3,4).astype('float32'))
    cmat2 = KCM.compress(mat1, 'auto')
    mat2 = cmat2.to_matrix()
    
    assert_allclose(mat1.data, mat2.data, atol=0.025)

    mat1 = KM(create_matrix_int().astype('float32'))
    cmat2 = KCM.compress(mat1, '2byte-signed-integer')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=0.75)

    mat1 = KM(create_matrix_uint().astype('float32'))
    cmat2 = KCM.compress(mat1, '1byte-unsigned-integer')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=1.5)


    mat1 = KM(create_matrix_01().astype('float32'))
    cmat2 = KCM.compress(mat1, '1byte-0-1')
    mat2 = cmat2.to_matrix()

    print(np.max(np.abs(mat1.data-mat2.data)))
    assert_allclose(mat1.data, mat2.data, atol=0.025)


    
def test_kcm_read_write():
    
    file_path = output_dir + '/kcm.mat'

    mat1 = KCM.compress(create_matrix().astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        mat2 = KCM.read(f, binary=False)

    assert_allclose(np.frombuffer(mat1.data, dtype=np.uint8),
                    np.frombuffer(mat2.data, dtype=np.uint8), atol=5)

    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        mat2 = KCM.read(f, binary=True)

    assert_allclose(np.frombuffer(mat1.data, dtype=np.uint8),
                    np.frombuffer(mat2.data, dtype=np.uint8))

    with open(file_path, 'rb') as f:
        mat2 = KCM.read(f, binary=True, row_offset=5)
    print(mat1.to_ndarray())
    print(mat2.to_ndarray())
    assert_allclose(mat1.to_ndarray()[5:], mat2.to_ndarray())

    with open(file_path, 'rb') as f:
        mat2 = KCM.read(f, binary=True, row_offset=4, num_rows=4)

    assert_allclose(mat1.to_ndarray()[4:8], mat2.to_ndarray())



def test_kcm_read_shape():
    
    file_path = output_dir + '/kcm.mat'

    mat1 = KCM.compress(create_matrix(10, 4).astype('float32'))
    with open(file_path, 'w') as f:
        mat1.write(f, binary=False)
    with open(file_path, 'rb') as f:
        assert KCM.read_shape(f, binary=False) == (10, 4)


    with open(file_path, 'wb') as f:
        mat1.write(f, binary=True)
    with open(file_path, 'rb') as f:
        assert KCM.read_shape(f, binary=True) == (10, 4)



def test_kcm_getbuild_data_attrs():

    mat1 = KM(create_matrix().astype('float32'))
    cmat2 = KCM.compress(mat1, 'auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data, attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data, mat3.data)


    cmat2 = KCM.compress(mat1, '2byte-auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data, attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data, mat3.data)


    cmat2 = KCM.compress(mat1, '1byte-auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data, attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data, mat3.data)


def test_kcm_getbuild_data_attrs_slice():

    mat1 = KM(create_matrix().astype('float32'))
    cmat2 = KCM.compress(mat1, 'auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data[2:7], attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data[2:7], mat3.data)


    cmat2 = KCM.compress(mat1, '2byte-auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data[2:7], attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data[2:7], mat3.data)


    cmat2 = KCM.compress(mat1, '1byte-auto')
    mat2 = cmat2.to_matrix()
    data, attrs = cmat2.get_data_attrs()
    cmat3 = KCM.build_from_data_attrs(data[2:7], attrs)
    mat3 = cmat3.to_matrix()

    assert_allclose(mat2.data[2:7], mat3.data)

    


if __name__ == '__main__':
    pytest.main([__file__])

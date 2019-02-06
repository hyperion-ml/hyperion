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

from hyperion.utils.trial_key import TrialKey
from hyperion.utils.trial_ndx import TrialNdx

output_dir = './tests/data_out/utils/trial'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        
def create_key(key_file='./tests/data_in/core-core_det5_key.h5'):

    key = TrialKey.load(key_file)
    key.sort()
    return key


def test_copy():

    key1 = create_key()
    key2 = key1.copy()

    key2.model_set[0] = 'm1'
    key2.tar[:] = 0
    assert(np.any(key1.model_set != key2.model_set))
    assert(np.any(key1.tar != key2.tar))


def test_merge():

    key1 = create_key()
    key2 = TrialKey(key1.model_set[:10], key1.seg_set,
                    key1.tar[:10,:], key1.non[:10,:])
    key3 = TrialKey(key1.model_set[5:], key1.seg_set,
                    key1.tar[5:,:], key1.non[5:,:])
    key4 = TrialKey.merge([key2, key3])
    assert(key1 == key4)

    key2 = TrialKey(key1.model_set, key1.seg_set[:10],
                    key1.tar[:,:10], key1.non[:,:10])
    key3 = TrialKey(key1.model_set, key1.seg_set[5:],
                    key1.tar[:,5:],  key1.non[:,5:])
    key4 = TrialKey.merge([key2, key3])
    assert(key1 == key4)


def test_filter():

    key1 = create_key()
    key2 = TrialKey(key1.model_set[:5], key1.seg_set[:10],
                    key1.tar[:5,:10], key1.non[:5,:10])
    key3 = key1.filter(key2.model_set, key2.seg_set, keep=True)
    assert(key2 == key3)


def test_split():

    key1 = create_key()
    num_parts=3
    key_list = []
    for i in xrange(num_parts):
        for j in xrange(num_parts):
            key_ij = key1.split(i+1, num_parts, j+1, num_parts)
            key_list.append(key_ij)
    key2 = TrialKey.merge(key_list)
    assert(key1 == key2)


def test_to_ndx():

    key1 = create_key()

    ndx1 = key1.to_ndx()
    ndx1.validate()

def test_load_save():

    key1 = create_key()
    file_h5 = output_dir + '/test.h5'
    key1.save(file_h5)
    key3 = TrialKey.load(file_h5)
    assert(key1 == key3)

    file_txt = output_dir + '/test.txt'
    key3.tar[0, :] = True
    key3.non[:, 0] = True
    key3.save(file_txt)
    key2 = TrialKey.load(file_txt)
    assert(key3 == key2)


if __name__ == '__main__':
    pytest.main([__file__])

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

from hyperion.utils.trial_ndx import TrialNdx

output_dir = './tests/data_out/utils/trial'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        
def create_ndx(ndx_file='./tests/data_in/core-core_det5_ndx.h5'):

    ndx = TrialNdx.load(ndx_file)
    ndx.sort()
    return ndx


def test_copy():

    ndx1 = create_ndx()
    ndx2 = ndx1.copy()

    ndx2.model_set[0] = 'm1'
    ndx2.trial_mask[:] = 0
    assert(np.any(ndx1.model_set != ndx2.model_set))
    assert(np.any(ndx1.trial_mask != ndx2.trial_mask))


def test_merge():

    ndx1 = create_ndx()
    ndx2 = TrialNdx(ndx1.model_set[:10], ndx1.seg_set,
                    ndx1.trial_mask[:10,:])
    ndx3 = TrialNdx(ndx1.model_set[5:], ndx1.seg_set,
                    ndx1.trial_mask[5:,:])
    ndx4 = TrialNdx.merge([ndx2, ndx3])
    assert(ndx1 == ndx4)

    ndx2 = TrialNdx(ndx1.model_set, ndx1.seg_set[:10],
                    ndx1.trial_mask[:,:10])
    ndx3 = TrialNdx(ndx1.model_set, ndx1.seg_set[5:],
                    ndx1.trial_mask[:,5:])
    ndx4 = TrialNdx.merge([ndx2, ndx3])
    assert(ndx1 == ndx4)


def test_filter():

    ndx1 = create_ndx()
    ndx2 = TrialNdx(ndx1.model_set[:5], ndx1.seg_set[:10],
                    ndx1.trial_mask[:5,:10])
    ndx3 = ndx1.filter(ndx2.model_set, ndx2.seg_set, keep=True)
    assert(ndx2 == ndx3)


def test_split():

    ndx1 = create_ndx()
    
    num_parts=3
    ndx_list = []
    for i in xrange(num_parts):
        for j in xrange(num_parts):
            ndx_ij = ndx1.split(i+1, num_parts, j+1, num_parts)
            ndx_list.append(ndx_ij)
    ndx2 = TrialNdx.merge(ndx_list)
    assert(ndx1 == ndx2)


def test_load_save():

    ndx1 = create_ndx()
    file_h5 = output_dir + '/test.h5'
    ndx1.save(file_h5)
    ndx3 = TrialNdx.load(file_h5)
    assert(ndx1 == ndx3)
    
    file_txt = output_dir + '/test.txt'
    ndx3.trial_mask[0, :] = True
    ndx3.trial_mask[:, 0] = True
    ndx3.save(file_txt)
    ndx2 = TrialNdx.load(file_txt)
    assert(ndx3 == ndx2)


if __name__ == '__main__':
    pytest.main([__file__])

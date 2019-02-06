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
from hyperion.utils.trial_scores import TrialScores

output_dir = './tests/data_out/utils/trial'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def create_scores(key_file='./tests/data_in/core-core_det5_key.h5'):

    key = TrialKey.load(key_file)

    mask = np.logical_or(key.tar, key.non)
    scr1 = TrialScores(key.model_set, key.seg_set,
                       np.random.normal(size=key.tar.shape)*mask,
                       mask)
    return scr1, key


def test_copy_sort_align():

    scr1, key = create_scores()
    scr2=scr1.copy()
    scr2.sort()
    assert(scr2 != scr1)
    scr3 = scr2.align_with_ndx(key)
    assert(scr1 == scr3)
    
    scr1.sort()
    scr2 = scr1.copy()
    
    scr2.model_set[0] = 'm1'
    scr2.score_mask[:] = 0
    assert(np.any(scr1.model_set != scr2.model_set))
    assert(np.any(scr1.score_mask != scr2.score_mask))


def test_merge():

    scr1 = create_scores()[0]
    scr1.sort()
    
    scr2 = TrialScores(scr1.model_set[:10], scr1.seg_set,
                       scr1.scores[:10,:], scr1.score_mask[:10,:])
    scr3 = TrialScores(scr1.model_set[10:], scr1.seg_set,
                       scr1.scores[10:,:], scr1.score_mask[10:,:])
    scr4 = TrialScores.merge([scr2, scr3])
    assert(scr1 == scr4)
    
    scr2 = TrialScores(scr1.model_set, scr1.seg_set[:10],
                       scr1.scores[:,:10], scr1.score_mask[:,:10])
    scr3 = TrialScores(scr1.model_set, scr1.seg_set[10:],
                       scr1.scores[:,10:], scr1.score_mask[:,10:])
    scr4 = TrialScores.merge([scr2, scr3])
    assert(scr1 == scr4)


def test_filter():

    scr1 = create_scores()[0]
    scr1.sort()

    scr2 = TrialScores(scr1.model_set[:5], scr1.seg_set[:10],
                       scr1.scores[:5,:10], scr1.score_mask[:5,:10])
    scr3 = scr1.filter(scr2.model_set, scr2.seg_set, keep=True)
    assert(scr2 == scr3)


def test_split():

    scr1 = create_scores()[0]
    scr1.sort()

    num_parts=3
    scr_list = []
    for i in xrange(num_parts):
        for j in xrange(num_parts):
            scr_ij = scr1.split(i+1, num_parts, j+1, num_parts)
            scr_list.append(scr_ij)
    scr2 = TrialScores.merge(scr_list)
    assert(scr1 == scr2)


def test_transform():

    scr1 = create_scores()[0]
    scr1.sort()
    
    f = lambda x: 3*x + 1
    scr2 = scr1.copy()
    scr2.score_mask[0,0] = True
    scr2.score_mask[0,1] = False
    scr4 = scr2.copy()
    scr4.transform(f)
    assert(scr4.scores[0,0] == 3*scr1.scores[0,0] + 1)
    assert(scr4.scores[0,1] == scr1.scores[0,1])


def test_get_tar_non():

    scr1, key = create_scores()

    scr2 = scr1.align_with_ndx(key)
    key2 = key.copy()
    scr2.score_mask[:] = False
    scr2.score_mask[0,0] = True
    scr2.score_mask[0,1] = True
    scr2.scores[0,0] = 1
    scr2.scores[0,1] = -1
    key2.tar[:] = False
    key2.non[:] = False
    key2.tar[0,0] = True
    key2.non[0,1] = True
    [tar, non] = scr2.get_tar_non(key2)
    assert(np.all(tar==[1]))
    assert(np.all(non==[-1]))


def test_set_missing_to_value():

    scr1, key = create_scores()

    scr2 = scr1.align_with_ndx(key)
    key2 = key.copy()
    scr2.score_mask[:] = False
    scr2.score_mask[0,0] = True
    scr2.score_mask[0,1] = True
    scr2.scores[0,0] = 1
    scr2.scores[0,1] = -1
    key2.tar[:] = False
    key2.non[:] = False
    key2.tar[0,0] = True
    key2.non[0,1] = True

    scr2.score_mask[0,0] = False
    scr4 = scr2.set_missing_to_value(key2, -10)
    assert(scr4.scores[0,0] == -10)


def test_load_save():

    scr1 = create_scores()[0]
    scr1.sort()
    
    file_h5 = output_dir + '/test.h5'
    scr1.save(file_h5)
    scr2 = TrialScores.load(file_h5)
    assert(scr1 == scr2)
    
    file_txt = output_dir + '/test.txt'
    scr1.score_mask[:, :] = False
    scr1.score_mask[0, :] = True
    scr1.score_mask[:, 0] = True
    scr1.scores[scr1.score_mask==False]=0
    scr1.save(file_txt)
    scr2 = TrialScores.load(file_txt)
          
    assert(scr1 == scr2)


if __name__ == '__main__':
    pytest.main([__file__])

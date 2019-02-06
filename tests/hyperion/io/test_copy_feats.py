"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange

import pytest

from hyperion.io.copy_feats import CopyFeats as CF

input_prefix = './tests/data_in/ark/'
feat_scp_b = 'scp:./tests/data_in/ark/feat_b.scp'
feat_ark_b = ['ark:./tests/data_in/ark/feat%d_b.ark' % i for i in xrange(1,3)]
feat_both_ho = 'h5,scp:./tests/data_out/h5/feat_cp.h5,./tests/data_out/h5/feat_cp.scp'


def test_copy_feats():
    CF(feat_scp_b, feat_both_ho,
       path_prefix=input_prefix, compress=True)


def test_merge_feats():
    CF(feat_ark_b, feat_both_ho,
       path_prefix=input_prefix, compress=True)


def test_split_feats():
    CF(feat_scp_b, feat_both_ho,
       path_prefix=input_prefix, compress=True,
       part_idx=2, num_parts=3)


    
if __name__ == '__main__':
    pytest.main([__file__])

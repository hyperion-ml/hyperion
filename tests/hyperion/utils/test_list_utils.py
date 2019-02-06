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

from hyperion.utils.list_utils import *

def create_lists():

    list1 = ['2', '2', '1', '2', '3', '3', '4', '4', '4', '4']
    list2 = ['0', '1', '2', '20']
    return list1, list2


def test_ismember():

    list1, list2 = create_lists()
    f, loc = ismember(list2, list1)
    assert(np.all(loc == [np.iinfo(np.int32).min, 2, 0, np.iinfo(np.int32).min]))
    assert(np.all(f == [False, True, True, False]))

    
def test_sort():

    list1, list2 = create_lists()
    list1_s, idx = sort(list1, return_index=True)
    assert(np.all(list1_s == ['1', '2', '2', '2', '3', '3', '4', '4', '4', '4']))

    
def test_unique():

    list1, list2 = create_lists()
    list1_u, i_a, i_b, = np.unique(list1, True, True)
    assert(np.all(list1_u == ['1', '2', '3', '4']))
    assert(np.all(i_a == [2,  0, 4, 6]))
    assert(np.all(i_b == [1, 1, 0, 1, 2, 2, 3, 3, 3, 3]))

    
def test_intersect():

    list1, list2 = create_lists()
    list1_u = np.unique(list1)
    list_i, i_a, i_b = intersect(list2, list1_u, return_index=True)
    assert(np.all(list_i == ['1', '2']))
    assert(np.all(i_a == [1, 2]))
    assert(np.all(i_b == [0, 1]))

    
def test_setdiff():

    list1, list2 = create_lists()
    list1_d = np.setdiff1d(list1, list2)
    assert(np.all(list1_d == ['3', '4']))
    list2_d = np.setdiff1d(list2, list1)
    assert(np.all(list2_d == ['0', '20']))



def test_split():

    list1, list2 = create_lists()
    list_s, loc = split_list(list1, 1, 3)
    print(list1)
    print(list_s)
    # ['2', '2', '1', '2', '3', '3', '4', '4', '4', '4']
    assert(np.all(list_s == ['2', '2', '1']))
    assert(np.all(loc == [0, 1, 2]))
    
    list_s, loc = split_list(list1, 2, 3)
    assert(np.all(list_s == ['2', '3', '3']))
    assert(np.all(loc == [3, 4, 5]))

    list_s, loc = split_list(list1, 3, 3)
    assert(np.all(list_s == ['4', '4', '4', '4']))
    assert(np.all(loc == [i for i in xrange(6, 10)]))


    
def test_split_by_key():

    list1, list2 = create_lists()
    list_s, loc = split_list_group_by_key(list1, 1, 3)
    assert(np.all(list_s == ['1']))
    assert(np.all(loc == [2]))
    
    list_s, loc = split_list_group_by_key(list1, 2, 3)
    assert(np.all(list_s == ['2', '2', '2']))
    assert(np.all(loc == [0, 1, 3]))

    list_s, loc = split_list_group_by_key(list1, 3, 3)
    assert(np.all(list_s == ['3', '3', '4', '4', '4', '4']))
    assert(np.all(loc == [i for i in xrange(4, 10)]))


if __name__ == '__main__':
    pytest.main([__file__])

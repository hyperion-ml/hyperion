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

from hyperion.utils.scp_list import SCPList

output_dir = './tests/data_out/utils/scp_list'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def create_scp():
    
    key = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
    file_path = np.arange(len(key)).astype('U')
    scp = SCPList(key, file_path)
    scp.sort()
    return scp


def create_scp_with_offset():
    
    key = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
    file_path = np.arange(len(key)).astype('U')
    offset = np.arange(len(key), dtype=np.int32)*10
    scp = SCPList(key, file_path, offset)
    scp.sort()
    return scp


def create_scp_with_offset_range():
    
    key = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
    file_path = np.arange(len(key)).astype('U')
    offset = np.arange(len(key), dtype=np.int32)*10
    range_spec = np.zeros((len(key), 2), dtype=np.int32)
    range_spec[3:,0] = 5
    range_spec[10:,1] = 10 
    scp = SCPList(key, file_path, offset, range_spec)
    scp.sort()
    return scp


def test_cmp():
    scp1 = create_scp()
    scp2 = create_scp()
    assert scp1 == scp2

    scp1 = create_scp_with_offset()
    scp2 = create_scp_with_offset()
    assert scp1 == scp2

    scp1 = create_scp_with_offset_range()
    scp2 = create_scp_with_offset_range()
    assert scp1 == scp2

    scp1 = SCPList([],[])
    scp2 = SCPList([],[])
    assert scp1 == scp2
    

def test_save_load():
    file_txt = './tests/data_out/list.scp'
    scp1 = create_scp()
    scp1.save(file_txt)
    scp2 = SCPList.load(file_txt)
    assert scp1 == scp2

    file_txt = './tests/data_out/list_offset.scp'
    scp1 = create_scp_with_offset()
    scp1.save(file_txt)
    scp2 = SCPList.load(file_txt)
    assert scp1 == scp2

    file_txt = './tests/data_out/list_offsetrange.scp'
    scp1 = create_scp_with_offset_range()
    scp1.save(file_txt)
    scp2 = SCPList.load(file_txt)
    assert scp1 == scp2
    

def test_split_merge():
    scp1 = create_scp()

    num_parts=3
    scp_list = []
    for i in xrange(num_parts):
        scp_i = scp1.split(i+1, num_parts)
        scp_list.append(scp_i)

    assert scp_list[0].len() == 1
    assert scp_list[1].len() == 10
    assert scp_list[2].len() == 5

    scp2 = SCPList.merge(scp_list)
    assert scp1 == scp2

    
    scp1 = create_scp_with_offset()

    num_parts=3
    scp_list = []
    for i in xrange(num_parts):
        scp_i = scp1.split(i+1, num_parts)
        scp_list.append(scp_i)

    assert scp_list[0].len() == 1
    assert scp_list[1].len() == 10
    assert scp_list[2].len() == 5

    scp2 = SCPList.merge(scp_list)
    assert scp1 == scp2


    scp1 = create_scp_with_offset_range()

    num_parts=3
    scp_list = []
    for i in xrange(num_parts):
        scp_i = scp1.split(i+1, num_parts)
        scp_list.append(scp_i)

    assert scp_list[0].len() == 1
    assert scp_list[1].len() == 10
    assert scp_list[2].len() == 5

    scp2 = SCPList.merge(scp_list)
    assert scp1 == scp2


    
def test_filter():
    filter_key = ['spk2', 'spk10']

    scp1 = create_scp()
    scp2 = scp1.filter(filter_key)
    
    f = np.zeros(len(scp1.key), dtype='bool')
    f[1:13] = True
    scp3 = SCPList(scp1.key[f], scp1.file_path[f])
        
    assert scp2 == scp3

    scp1 = create_scp_with_offset_range()
    scp2 = scp1.filter(filter_key)
    
    f = np.zeros(len(scp1.key), dtype='bool')
    f[1:13] = True
    scp3 = SCPList(scp1.key[f], scp1.file_path[f],
                   scp1.offset[f], scp1.range_spec[f])
    print(scp2.__dict__)
    print(scp3.__dict__)
    assert scp2 == scp3

    
    filter_key=[]
    scp2 = scp1.filter(filter_key)
    scp3 = SCPList([],[])
    assert scp2 == scp3



def test_shuffle():

    scp1 = create_scp()
    scp1.shuffle()

    scp1 = create_scp_with_offset()
    scp1.shuffle()

    scp1 = create_scp_with_offset_range()
    scp2 = scp1.copy()
    index = scp1.shuffle()
    scp2 = scp2.filter_index(index)

    assert scp1 == scp2


def test_getitem():

    scp1 = create_scp()
    assert scp1[1][1] == '6'

    assert scp1['spk1'][0] == '0'
    assert scp1['spk10'][0] == '15'

    assert 'spk1' in scp1

    
if __name__ == '__main__':
    pytest.main([__file__])

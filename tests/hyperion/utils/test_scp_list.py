
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import pytest
import numpy as np

from hyperion.utils.scp_list import SCPList


def create_scp():
    
    key = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
    file_path = np.arange(len(key)).astype('U')
    scp = SCPList(key, file_path)
    scp.sort()
    return scp

def test_cmp():
    scp1 = create_scp()
    scp2 = create_scp()
    assert(scp1 == scp2)

    scp1 = SCPList([],[])
    scp2 = SCPList([],[])
    assert(scp1 == scp2)
    

def test_save_load():
    file_txt = './tests/data_out/list.scp'
    scp1 = create_scp()
    scp1.save(file_txt)
    scp2 = SCPList.load(file_txt)
    assert(scp1 == scp2)

def test_split_merge():
    scp1 = create_scp()

    num_parts=3
    scp_list = []
    for i in xrange(num_parts):
        scp_i = scp1.split(i+1, num_parts)
        scp_list.append(scp_i)

    assert(scp_list[0].len() == 1)
    assert(scp_list[1].len() == 10)
    assert(scp_list[2].len() == 5)

    scp2 = SCPList.merge(scp_list)
    assert(scp1 == scp2)

def test_filter():
    filter_key = ['spk2', 'spk10']
    scp1 = create_scp()
    scp2 = scp1.filter(filter_key)
    
    f = np.zeros(len(scp1.key), dtype='bool')
    f[1:13] = True
    scp3 = SCPList(scp1.key[f], scp1.file_path[f])
        
    assert(scp2 == scp3)

    filter_key=[]
    scp2 = scp1.filter(filter_key)
    scp3 = SCPList([],[])
    assert(scp2 == scp3)



if __name__ == '__main__':
    pytest.main([__file__])

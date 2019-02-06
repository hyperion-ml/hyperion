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

from hyperion.utils.utt2info import Utt2Info

output_dir = './tests/data_out/utils/utt2info'
if not os.path.exists(output_dir):
     os.makedirs(output_dir)
        

def create_utt2info():
    
    spk_ids = ['spk1']+['spk2']*2+['spk3']*3+['spk10']*10
    key = np.arange(len(spk_ids)).astype('U')
    u2i = Utt2Info.create(key, spk_ids)
    u2i.sort(field=1)
    return u2i



def test_cmp():
    u2i1 = create_utt2info()
    u2i2 = create_utt2info()
    assert u2i1 == u2i2

    u2i1 = Utt2Info.create([],[])
    u2i2 = Utt2Info.create([],[])
    assert u2i1 == u2i2


def test_sort():

    u2i = create_utt2info()
    spk_ids = ['spk1']+['spk10']*10+['spk2']*2+['spk3']*3
    assert np.all(u2i.info == np.asarray(spk_ids))


def test_len():
    u2i = create_utt2info()
    assert len(u2i) == 16
    assert u2i.len() == 16


def test_get_index():
    u2i = create_utt2info()
    assert u2i.get_index('1') == 11
    

def test_contains():
    u2i = create_utt2info()
    assert '10' in u2i
    assert not('100' in u2i)



def test_getitem():

    u2i1 = create_utt2info()
    assert u2i1[1][1] == 'spk10'

    assert u2i1['0'] == 'spk1'


    
def test_save_load():
    file_txt = output_dir + '/utt2spk'
    u2i1 = create_utt2info()
    u2i1.save(file_txt)
    u2i2 = Utt2Info.load(file_txt)
    print(u2i1.utt_info)
    print(u2i2.utt_info)
    print(np.asarray(u2i1.utt_info['key']) == np.asarray(u2i2.utt_info['key']))
    print(np.asarray(u2i1.utt_info['key']))
    print(np.asarray(u2i2.utt_info['key']))
    assert u2i1 == u2i2
    

    
def test_split_merge():
    u2i1 = create_utt2info()

    num_parts=3
    u2i_list = []
    for i in xrange(num_parts):
        u2i_i = u2i1.split(i+1, num_parts, group_by_field=1)
        u2i_list.append(u2i_i)

    assert u2i_list[0].len() == 1
    assert u2i_list[1].len() == 10
    assert u2i_list[2].len() == 5

    u2i2 = Utt2Info.merge(u2i_list)
    assert u2i1 == u2i2


    
def test_filter():
    filter_key = ['0', '1', '2']

    u2i1 = create_utt2info()
    u2i2 = u2i1.filter(filter_key)
    
    idx = [0, 11, 12]
    u2i3 = Utt2Info.create(u2i1.key[idx], u2i1.info[idx])
        
    assert u2i2 == u2i3


    
def test_filter_info():
    filter_key = ['spk2', 'spk10']

    u2i1 = create_utt2info()
    u2i2 = u2i1.filter_info(filter_key)
    
    f = np.zeros(len(u2i1.key), dtype='bool')
    f[1:13] = True
    u2i3 = Utt2Info.create(u2i1.key[f], u2i1.info[f])

    print(u2i2.utt_info)
    print(u2i3.utt_info)
        
    assert u2i2 == u2i3


def test_filter_index():
    filter_key = [0, 11, 12]

    u2i1 = create_utt2info()
    u2i2 = u2i1.filter_index(filter_key)
    
    idx = [0, 11, 12]
    u2i3 = Utt2Info.create(u2i1.key[idx], u2i1.info[idx])

    assert u2i2 == u2i3



def test_shuffle():

    u2i1 = create_utt2info()
    u2i2 = u2i1.copy()
    index = u2i1.shuffle()
    u2i2 = u2i2.filter_index(index)

    assert u2i1 == u2i2


    
if __name__ == '__main__':
    pytest.main([__file__])

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

from hyperion.io.rw_specifiers import *

output_dir = './tests/data_out/io'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def test_rspecifier():

    rs1 = RSpecifier(RSpecType.ARCHIVE, 'file.h5', ArchiveType.H5)
    rs2 = RSpecifier.create('file.h5')
    assert rs1 == rs2

    rs2 = RSpecifier.create('h5:file.h5')
    assert rs1 == rs2

    rs1 = RSpecifier(RSpecType.ARCHIVE, 'file.ark', ArchiveType.ARK)
    rs2 = RSpecifier.create('ark:file.ark')
    assert rs1 == rs2

    rs1 = RSpecifier(RSpecType.ARCHIVE, 'file.ark', ArchiveType.ARK,
                     True, True, True, True, True)
    rs2 = RSpecifier.create('ark,o,s,cs,p,bg:file.ark')
    assert rs1 == rs2

    file_path = output_dir + '/file.scp'
    with open(file_path, 'w') as f:
        f.write('key file1:0\n')
    rs1 = RSpecifier(RSpecType.SCRIPT, file_path, ArchiveType.ARK)
    rs2 = RSpecifier.create('scp:' + file_path)
    assert rs1 == rs2

    with open(file_path, 'w') as f:
        f.write('key file1\n')
    rs1 = RSpecifier(RSpecType.SCRIPT, file_path, ArchiveType.H5)
    rs2 = RSpecifier.create('scp:' + file_path)
    assert rs1 == rs2


def test_wspecifier():

    rs1 = WSpecifier(WSpecType.ARCHIVE, 'file.h5', None, ArchiveType.H5)
    rs2 = WSpecifier.create('file.h5')
    assert rs1 == rs2

    rs2 = WSpecifier.create('h5:file.h5')
    assert rs1 == rs2

    rs1 = WSpecifier(WSpecType.ARCHIVE, 'file.ark', None, ArchiveType.ARK)
    rs2 = WSpecifier.create('ark:file.ark')
    assert rs1 == rs2

    rs1 = WSpecifier(WSpecType.ARCHIVE, 'file.ark', None, ArchiveType.ARK)
    rs2 = WSpecifier.create('ark,b,nf:file.ark')
    assert rs1 == rs2

    rs1 = WSpecifier(WSpecType.ARCHIVE, 'file.ark', None, ArchiveType.ARK)
    rs2 = WSpecifier.create('ark,b,nf:file.ark')
    assert rs1 == rs2

    
    rs1 = WSpecifier(WSpecType.ARCHIVE, 'file.ark', None, ArchiveType.ARK,
                     False, True, True)
    rs2 = WSpecifier.create('ark,t,f,p:file.ark')
    assert rs1 == rs2
    
    rs1 = WSpecifier(WSpecType.SCRIPT, None, 'file.scp', None)
    rs2 = WSpecifier.create('scp:file.scp')
    assert rs1 == rs2

    rs1 = WSpecifier(WSpecType.BOTH, 'file.ark', 'file.scp',
                     ArchiveType.ARK,False)
    rs2 = WSpecifier.create('ark,t,scp:file.ark,file.scp')
    assert rs1 == rs2

    rs1 = WSpecifier(WSpecType.BOTH, 'file.h5', 'file.scp',
                     ArchiveType.H5)
    rs2 = WSpecifier.create('h5,scp:file.h5,file.scp')
    assert rs1 == rs2


if __name__ == '__main__':
    pytest.main([__file__])

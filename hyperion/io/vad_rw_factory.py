"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import

import logging

from .rw_specifiers import ArchiveType, WSpecifier, RSpecifier, WSpecType, RSpecType
from .bin_vad_reader import BinVADReader as BVR



class VADReaderFactory(object):

    @staticmethod
    def create(rspecifier, path_prefix=None, scp_sep=' '):
        if isinstance(rspecifier, str):
            rspecifier = RSpecifier.create(rspecifier)
        logging.debug(rspecifier.__dict__)
        if rspecifier.spec_type ==  RSpecType.ARCHIVE:
            if (rspecifier.archive_type == ArchiveType.H5 or 
                rspecifier.archive_type == ArchiveType.ARK):
                return BVR(rspecifier, path_prefix, scp_sep)
        else:
            if (rspecifier.archive_type == ArchiveType.H5 or 
                rspecifier.archive_type == ArchiveType.ARK):
                return BVR(rspecifier, path_prefix, scp_sep)



    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'path_prefix')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        
        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        if prefix is None:
            p1 = '--'
            p2 = ''
        else:
            p1 = '--' + prefix + '-'
            p2 = prefix + '_'
            
        parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default=' ',
                            help=('scp file field separator'))
        parser.add_argument(p1+'path-prefix', dest=(p2+'path_prefix'), default=None,
                            help=('scp file_path prefix'))




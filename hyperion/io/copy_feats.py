"""
Class to convert between feature file formats
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange
from six import string_types

from ..utils.kaldi_matrix import compression_methods
from .data_rw_factory import DataWriterFactory as DWF
from .data_rw_factory import SequentialDataReaderFactory as DRF


class CopyFeats(object):

    def __init__(self, input_spec, output_spec, path_prefix=None,
                 compress=False, compression_method='auto',
                 scp_sep=' ', part_idx=1, num_parts=1, chunk_size=1):
        
        if isinstance(input_spec, string_types):
            input_spec = [input_spec]

        assert not(num_parts>1 and len(input_spec)>1), (
            'Merging and splitting at the same time is not supported')

        with DWF.create(output_spec,
                        compress=compress, compression_method=compression_method,
                        scp_sep=scp_sep) as writer:
        
            for rspec in input_spec:
                with DRF.create(rspec, path_prefix=path_prefix, scp_sep=scp_sep,
                         part_idx=part_idx, num_parts=num_parts) as reader:
                    while not reader.eof():
                        key, data = reader.read(chunk_size)
                        if len(key) == 0:
                            break
                        writer.write(key, data)
            

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'path_prefix', 'part_idx', 'num_parts')
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
        parser.add_argument(p1+'part-idx', dest=(p2+'part_idx'), type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))
        parser.add_argument(p1+'num-parts', dest=(p2+'num_parts'), type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))

        parser.add_argument('--compress', dest='compress', default=False, action='store_true')
        parser.add_argument('--compression-method', dest='compression_method', default='auto',
                            choices=compression_methods)

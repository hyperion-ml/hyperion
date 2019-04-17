"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from six.moves import xrange
from six import string_types

import logging

from ..utils.kaldi_matrix import compression_methods
from .data_rw_factory import DataWriterFactory as DWF
from .data_rw_factory import SequentialDataReaderFactory as DRF


class CopyFeats(object):
    """ Class to convet between Ark/hdf5 feature formats.
    """
    
    def __init__(self, input_spec, output_spec, path_prefix=None,
                 compress=False, compression_method='auto',
                 scp_sep=' ', part_idx=1, num_parts=1, chunk_size=1):
        """CopyFeats constructor, it executes the conversion.

        Args:
          input_spec: Kaldi style read specifier, e.g.:
                      file.h5
                      h5:file.h5
                      ark:file.ark
                      scp:file.scp

                      or list of specifiers, e.g.:
                      ['scp:file1.scp', 'scp:file2.scp']

                      if it is a list, it merges the input files.

           output_spec: Kaldi stype write specifier, e.g.:
                        file.h5
                        h5:file.h5
                        ark:file.ark
                        h5,scp:file.h5,file.scp
                        ark,scp:file.ark,file.scp

           path_prefix: If input_spec is a scp file, it pre-appends 
                        path_prefix string to the second column of 
                        the scp file. This is useful when data 
                        is read from a different directory of that 
                        it was created.
           compress: if True, it  compress the features (default: False).
           compression_method: Kaldi compression method:
                               {auto (default), speech_feat, 
                                2byte-auto, 2byte-signed-integer,
                                1byte-auto, 1byte-unsigned-integer, 1byte-0-1}.
           scp_sep: Separator for scp files (default ' ').
           part_idx: It splits the input into num_parts and writes only 
                     part part_idx, where part_idx=1,...,num_parts.
           num_parts: Number of parts to split the input data.
           chunk_size: When copying, it reads the input files in groups of 
                       chunk_size (default:1).
        """
        if isinstance(input_spec, string_types):
            input_spec = [input_spec]

        assert not(num_parts>1 and len(input_spec)>1), (
            'Merging and splitting at the same time is not supported')

        logging.info('opening output stream: %s' % (output_spec))
        with DWF.create(output_spec,
                        compress=compress, compression_method=compression_method,
                        scp_sep=scp_sep) as writer:
        
            for rspec in input_spec:
                logging.info('opening input stream: %s' % (rspec))
                with DRF.create(rspec, path_prefix=path_prefix, scp_sep=scp_sep,
                         part_idx=part_idx, num_parts=num_parts) as reader:
                    while not reader.eof():
                        key, data = reader.read(chunk_size)
                        if len(key) == 0:
                            break
                        logging.info('copying %d feat matrices' % (len(key)))
                        writer.write(key, data)
            

    @staticmethod
    def filter_args(prefix=None, **kwargs):
        """Extracts the relevant arguments for the CopyFeats object.
        
        Args:
          prefix: Prefix for the name of the argument.
          kwargs: Dictionary containing arguments for several classes.
        
        Returns:
          Dictionary with the relevant arguments to initialize the object.
        """
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('scp_sep', 'path_prefix', 'part_idx', 'num_parts')
        return dict((k, kwargs[p+k])
                    for k in valid_args if p+k in kwargs)

        
        
    @staticmethod
    def add_argparse_args(parser, prefix=None):
        """Adds arguments required to initialize the object to python
           argparse object.
        
        Args:
          parser: Python argparse object.
          prefix: Prefix for the argument names. The prefix is useful when you have
                  several objects of the same class in the program and you want to
                  initialize each of them with different arguments.
        """
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

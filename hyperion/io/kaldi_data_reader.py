"""
Class to read input and target features from .ark files.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import sys
import gzip, struct, re
from collections import OrderedDict
import numpy as np

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList

class KaldiDataReader(object):

      def __init__(self, file_path, input_dir=None, sep=' '):
            self.file_path = file_path
            self.cur_record=0
            
            scp = SCPList.load(file_path, sep=sep)
            if input_dir is None:
                  self.scp = OrderedDict((k, v) for (k, v) in zip(scp.key, scp.file_path))
            else:
                  input_dir+='/'
                  self.scp = OrderedDict((k, input_dir+v) for (k, v) in zip(scp.key, scp.file_path))

                  
      def read(self, keys=None, num_records=None, first_record=None, squeeze=False):
            if keys is None:
                  keys=list(self.scp.keys())
                  if first_record is not None:
                        self.cur_record = first_record
                        
                  if num_records is None:
                        keys = keys[self.cur_record:]
                        self.cur_record = len(keys)
                  else:
                        final_record = min(self.cur_record+num_records, len(keys))
                        keys = keys[self.cur_record:final_record]
                        self.cur_record = final_record

            X = []
            for i, key in enumerate(keys):
                  file_path = self.scp[key]
                  m = self._read_matrix(file_path)
                  if squeeze:
                        m = np.squeeze(m)
                  X.append(m)
                  
            return X, keys

      
      def reset(self):
            self.cur_record=0

      def eof(self):
            return self.cur_record == len(self.scp.keys())
      
      @staticmethod
      def _open(file_path, mode='rb'):
            try:
                  # separate offset from filename (optional),
                  offset = None
                  if re.search(':[0-9]+$', file_path):
                        (file_path, offset) = file_path.rsplit(':',1)

                  if file_path.split('.')[-1] == 'gz':
                        f = gzip.open(file_path, 'r')
                  else:
                        f = open(file_path, 'r')
                  if offset is not None:
                        f.seek(int(offset))
                  return f
            except TypeError:
                  return file_path

            
      @staticmethod
      def _read_matrix(f):
            f = KaldiDataReader._open(f)
            binary = f.read(2)
            if binary == b'\0B' :
                  mat = KaldiDataReader._read_bin_matrix(f)
            else:
                  assert(binary == ' [')
                  mat = KaldiDataReader._read_ascii_matrix(f)
            return mat

      
      @staticmethod
      def _read_bin_matrix(f):
            stype = f.read(3)
            dtype=None
            if stype == b'FM ': dtype = 'float32'
            if stype == b'DM ': dtype = 'float64'
            assert(dtype is not None)
            # Dimensions
            f.read(1)
            rows = struct.unpack('<i', f.read(4))[0]
            f.read(1)
            cols = struct.unpack('<i', f.read(4))[0]
            # Read whole matrix
            buf = fd.read(rows * cols * np.dtype(dtype).itemsize)
            vec = np.frombuffer(buf, dtype=dtype) 
            mat = np.reshape(vec,(rows,cols))
            return mat

      
      @staticmethod
      def _read_ascii_matrix(f):
            rows = []
            while 1:
                  line = f.readline()
                  if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
                  if len(line.strip()) == 0 : continue # skip empty line
                  arr = line.strip().split()
                  if arr[-1] != ']':
                        rows.append(np.array(arr,dtype='float32')) # not last line
                  else: 
                        rows.append(np.array(arr[:-1],dtype='float32')) # last line
                        mat = np.vstack(rows)
                        return mat

            
      

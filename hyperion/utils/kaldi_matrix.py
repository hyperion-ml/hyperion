"""
Function to write and read kaldi matrices
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import struct
import numpy as np

from ..hyp_defs import float_cpu
from ..utils.kaldi_io_funcs import *


class KaldiMatrix(object):
    def __init__(self, data):
        self.data=data


    def to_ndarray(self):
        return self.data
    

    @property
    def num_rows(self):
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]

    
    @property
    def num_cols(self):
        if self.data.ndim == 1:
            return self.data.shape[0]
        return self.data.shape[1]

    
    @classmethod
    def read(cls, f, binary, row_offset=0, num_rows=0, sequential_mode=True):
        if binary:
            peekval = peek(f, binary)
            if peekval == b'C':
                mat = KaldiCompressedMatrix.read(
                    f, binary, row_offset, num_rows, sequential_mode).to_ndarray()
                return cls(mat)
            token = read_token(f, binary)
            if token[0] == 'F' : dtype = 'float32'
            elif token[0] == 'D': dtype = 'float64'
            else:
                ValueError('Wrong token %s ' % token)
            if token[1] == 'V' : ndim = 1
            elif token[1] == 'M': ndim = 2
            else:
                ValueError('Wrong token %s ' % token)

            rows_left = 0
            if ndim == 2:
                total_rows = read_int32(f, binary)
                assert row_offset <= total_rows, (
                    'row_offset (%d) > num_rows (%d)' %
                    (row_offset, total_rows))
                total_rows -= row_offset
                if num_rows == 0:
                    num_rows = total_rows
                else:
                    assert num_rows <= total_rows, (
                        'requested rows (%d) > available rows (%d)' %
                        (num_rows, total_rows))
                    left_row = total_rows - num_rows
                    
            else:
                num_rows = 1
            num_cols = read_int32(f, binary)
            
            if row_offset > 0:
                f.seek(row_offset*num_cols*np.dtype(dtype).itemsize, 1)
            data = f.read(num_rows*num_cols*np.dtype(dtype).itemsize)
            if rows_left > 0 and sequential_mode:
                f.seek(rows_left*num_cols*np.dtype(dtype).itemsize, 1)
                
            vec = np.frombuffer(data, dtype=dtype)
            
            if ndim == 2:
                return cls(np.reshape(vec, (num_rows, num_cols)))
            return cls(vec)
        
        else:
            assert row_offset == 0, 'row offset not supported in text mode because it is inefficient'
            first_line = True
            rows = []
            is_vector = False
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode('ascii')
                    
                if len(line) == 0 :
                    raise BadInputFormat('EOF reading matrix') # eof, should not happen!
                if len(line.strip()) == 0 : continue # skip empty line

                arr = line.strip().split()
                if first_line:
                    if arr == '[]':
                        return np.array([], dtype='float32')
                    if arr[0] != '[':
                        raise ValueError('Wrong matrix format %s ' % line)
                    first_line = False
                    if len(arr) > 1:
                        is_vector = True
                        arr = arr[1:]
                    else:
                        continue
                    
                if arr[-1] != ']':
                    rows.append(np.array(arr, dtype='float32')) # not last line
                else: 
                    rows.append(np.array(arr[:-1], dtype='float32')) # last line
                    mat = np.vstack(rows)
                    if mat.shape[0] == 1 and is_vector:
                        mat = mat.ravel()
                    return cls(mat)
                
            return cls(np.array([], dtype='float32'))
            

        
    def write(self, f, binary):
        if binary:
            t1 = 'F' if self.data.dtype == np.float32 else 'D'
            t2 = 'M' if self.data.ndim == 2 else 'V'
            token = t1+t2
            write_token(f, binary, token)
            if self.data.ndim == 2:
                write_int32(f, binary, self.num_rows)
            write_int32(f, binary, self.num_cols)
            f.write(self.data.tobytes())
        else:
            if self.num_cols == 0:
                f.write(' [ ]\n')
            else:
                f.write(' [')
                if self.data.ndim == 1:
                    f.write(' ')
                    for j in xrange(self.num_cols):
                        f.write('%f ' % self.data[j])
                else:
                    for i in xrange(self.num_rows):
                        f.write('\n ')
                        for j in xrange(self.num_cols):
                            f.write('%f ' % self.data[i,j])
                f.write(']\n')


    @staticmethod
    def read_shape(f, binary, sequential_mode=True):
        if binary:
            peekval = peek(f, binary)
            if peekval == b'C':
                return KaldiCompressedMatrix.read_shape(f, binary, sequential_mode)
            token = read_token(f, binary)
            if token[0] == 'F' : dtype = 'float32'
            elif token[0] == 'D': dtype = 'float64'
            else:
                ValueError('Wrong token %s ' % token)
            if token[1] == 'V' : ndim = 1
            elif token[1] == 'M': ndim = 2
            else:
                ValueError('Wrong token %s ' % token)

            if ndim == 2:
                num_rows = read_int32(f, binary)
            else:
                num_rows = 1
            num_cols = read_int32(f, binary)
            if sequential_mode:
                f.seek(num_rows*num_cols*np.dtype(dtype).itemsize, 1)

            if ndim == 1:
                return (num_cols,)
            else:
                return (num_rows, num_cols)
        else:
            matrix = KaldiMatrix.read(f, binary, sequential_mode=sequential_mode)
            return matrix.data.shape

        
        
compression_methods = {'auto': 1,
                       'speech-feat': 2,
                       '2byte-auto': 3,
                       '2byte-signed-integer': 4,
                       '1byte-auto': 5,
                       '1byte-unsigned-integer': 6,
                       '1byte-0-1': 7}

compression_method2format = {'speech-feat': 1,
                             '2byte-auto': 2,
                             '2byte-signed-integer': 2,
                             '1byte-auto': 1,
                             '1byte-unsigned-integer': 1,
                             '1byte-0-1': 1}

    
class KaldiCompressedMatrix(object):
    def __init__(self, data=None):
        self.data = data
        self.data_format = 1
        self.min_value = 0
        self.data_range = 0
        self.num_rows = 0
        self.num_cols = 0
        
        if data is not None:
            self._unpack_header()
        # self.col_headers = col_headers


    def _unpack_header(self):
        h = struct.unpack('<iffii', self.data[:20])
        self.data_format = h[0]
        self.min_value = h[1]
        self.data_range = h[2]
        self.num_rows = h[3]
        self.num_cols = h[4]

        
    def _pack_header(self):
        return struct.pack('<iffii',
                           self.data_format,
                           self.min_value, self.data_range,
                           self.num_rows, self.num_cols)
        
        
    def scale(self, alpha):
        self.min_value *= alpha
        self.data_range *= alpha
        header = self._pack_header()
        self.data = header + self.data[20:]


        
    def _compute_global_header(self, mat, method):
        if method == 'auto':
            if mat.shape[0] > 8:
                method = 'speech-feat'
            else:
                method = '2byte-auto'
        self.data_format = compression_method2format[method]
        self.num_rows = mat.shape[0]
        self.num_cols = mat.shape[1]

        #now compute min_val and range
        if method == 'speech-feat' or method == '2byte-auto' or method == '1byte-auto':
            min_value = np.min(mat)
            max_value = np.max(mat)
            if max_value == min_value :
                max_value = min_value + 1 + np.abs(min_value)
            assert min_value-min_value==0 and max_value-max_value==0, 'cannot compress matrix with infs or nans'
            self.min_value = min_value
            self.data_range = max_value - min_value
            assert self.data_range > 0
        elif method == '2byte-signed-integer':
            self.min_value = -32768.0
            self.data_range = 65535.0
        elif method == '1byte-unsigned-integer':
            self.min_value = 0.0
            self.data_range = 255.0
        elif method == '1byte-0-1':
            self.min_value = 0.0
            self.data_range = 1.0
        else:
            raise ValueError(method)

        header = self._pack_header()
        return header


    
    @staticmethod
    def _get_read_info2(header, row_offset=0, num_rows=0):
        data_format, min_value, data_range, total_rows, num_cols = struct.unpack('<iffii', header)
        make_header = True if row_offset !=0 or num_rows != 0 else False
            
        rows_left = 0
        assert row_offset <= total_rows, (
            'row_offset (%d) > num_rows (%d)' %
            (row_offset, total_rows))
        total_rows -= row_offset
        if num_rows == 0:
            num_rows = total_rows
        else:
            assert num_rows <= total_rows, (
                'requested rows (%d) > available rows (%d)' %
                (num_rows, total_rows))
            rows_left = total_rows - num_rows
            
        bytes_col_header = 0
        if data_format == 1:
            bytes_col_header = num_cols*8
            bytes_offset = row_offset*num_cols
            bytes_data = num_rows*num_cols
            bytes_left = rows_left*num_cols
        elif data_format == 2:
            bytes_offset = 2*row_offset*num_cols
            bytes_data = 2*num_rows*num_cols
            bytes_left = 2*rows_left*num_cols
        else:
            bytes_offset = row_offset*num_cols
            bytes_data = num_rows*num_cols
            bytes_left = rows_left*num_cols

        if make_header:
            header = struct.pack(
                '<iffii', data_format, min_value, data_range,
                num_rows, num_cols)
            
        return header, bytes_col_header, bytes_offset, bytes_data, bytes_left


    @staticmethod
    def _get_read_info(header, row_offset=0, num_rows=0):
        data_format, min_value, data_range, total_rows, num_cols = struct.unpack('<iffii', header)
        make_header = True if row_offset !=0 or num_rows != 0 else False
            
        rows_left = 0
        assert row_offset <= total_rows, (
            'row_offset (%d) > num_rows (%d)' %
            (row_offset, total_rows))
        total_rows -= row_offset
        if num_rows == 0:
            num_rows = total_rows
        else:
            assert num_rows <= total_rows, (
                'requested rows (%d) > available rows (%d)' %
                (num_rows, total_rows))
            rows_left = total_rows - num_rows
            
        bytes_col_header = 0
        if data_format == 1:
            bytes_col_header = num_cols*8
            bytes_offset = row_offset
            bytes_data = num_rows
            bytes_left = rows_left
        elif data_format == 2:
            bytes_offset = 2*row_offset
            bytes_data = 2*num_rows
            bytes_left = 2*rows_left
        else:
            bytes_offset = row_offset
            bytes_data = num_rows
            bytes_left = rows_left

        if make_header:
            header = struct.pack(
                '<iffii', data_format, min_value, data_range,
                num_rows, num_cols)
            
        return header, num_cols, bytes_col_header, bytes_offset, bytes_data, bytes_left



    @staticmethod
    def _data_size(header):
        data_format, _, _, num_rows, num_cols = struct.unpack('<iffii', header)
        if data_format == 1:
            return len(header) + num_cols*(8+num_rows)
        elif data_format == 2:
            return len(header) + 2*num_rows*num_cols
        else:
            return len(header) + num_rows*num_cols


        
    @classmethod
    def compress(cls, mat, method='auto'):

        if isinstance(mat, KaldiMatrix):
            mat = mat.data

        M = cls()
        header = M._compute_global_header(mat, method)
        cols_header = bytes()
        data = bytes()
        if M.data_format == 1:
            for col in xrange(M.num_cols):
                col_header, col_data = M._compress_column(mat[:,col])
                cols_header += col_header
                data += col_data

        elif M.data_format == 2:
            data = M._float_to_uint16(mat).tobytes()
        else:
            data = M._float_to_uint8(mat).tobytes()

        M.data = header + cols_header + data
        return M
    
        

    def _float_to_uint16(self, mat):
        f = (mat.ravel() - self.min_value)/self.data_range
        f[f>1.0] = 1
        f[f<0.0] = 0
        return (f*65535 + 0.499).astype(np.uint16)


    
    def _float_to_uint8(self, mat):
        f = (mat.ravel() - self.min_value)/self.data_range
        f[f>1.0] = 1
        f[f<0.0] = 0
        return (f*255 + 0.499).astype(np.uint8)


    
    def _uint16_to_float(self, byte_data):
        return self.min_value + self.data_range * 1.52590218966964e-05 * np.fromstring(
            byte_data, dtype=np.uint16)
        


    def _uint8_to_float(self, byte_data):
        return self.min_value + self.data_range/255.0 * np.fromstring(
            byte_data, dtype=np.uint8)


    
    def _compute_column_header(self, v):
        one = np.uint16(1)
        if self.num_rows >= 5:
            quarter_nr = int(self.num_rows/4)
            v_sort = np.partition(v, (0, quarter_nr, 3*quarter_nr, -1))
            perc_0 = min(self._float_to_uint16(v_sort[0])[0], np.uint16(65532))
            perc_25 = min(max(self._float_to_uint16(v_sort[quarter_nr])[0], perc_0 + one), np.uint16(65533))
            perc_75 = min(max(self._float_to_uint16(v_sort[3*quarter_nr])[0], perc_25 + one), np.uint16(65534))
            perc_100 = max(self._float_to_uint16(v_sort[-1])[0], perc_75 + one)
        else:
            v_sort = np.sort(v)
            perc_0 = min(self._float_to_uint16(v_sort[0])[0], np.uint16(65532))
            if self.num_rows > 1:
                perc_25 = min(max(self._float_to_uint16(v_sort[1])[0], perc_0 + one), np.uint16(65533))
            else:
                perc_25 = perc_0 +1
            if self.num_rows > 2:
                perc_75 = min(max(self._float_to_uint16(v_sort[2])[0], perc_25 + one), np.uint16(65534))
            else:
                perc_75 = perc_25 + one

            if self.num_rows > 3:
                perc_100 = max(self._float_to_uint16(v_sort[3])[0], perc_75 + one)
            else:
                perc_100 = perc_75 + one
        return struct.pack('<HHHH', perc_0, perc_25, perc_75, perc_100)


    
    def _compress_column(self, v):
        col_header = self._compute_column_header(v)
        p0, p25, p75, p100 = self._uint16_to_float(col_header)
        return col_header, self._float_to_char(v, p0, p25, p75, p100).tobytes()


    
    def _uncompress_column(self, col_header, col_data):
        p0, p25, p75, p100 = self._uint16_to_float(col_header)
        return self._char_to_float(col_data, p0, p25, p75, p100)
        
    

    @staticmethod
    def _float_to_char(v, p0, p25, p75, p100):
        v_out = np.zeros(v.shape, dtype=np.int32)
        idx = v < p25
        f = (v[idx] - p0)/(p25-p0)
        c = (f*64+0.5).astype(np.int32)
        c[c<0] = 0
        c[c>64] = 64
        v_out[idx] = c
        idx = np.logical_and(v >= p25, v < p75)
        f = (v[idx] - p25)/(p75-p25)
        c = 64 + (f*128+0.5).astype(np.int32)
        c[c<64] = 64
        c[c>192] = 192
        v_out[idx] = c
        idx =v >= p75
        f = (v[idx] - p75)/(p100-p75)
        c = 192 + (f*63+0.5).astype(np.int32)
        c[c<192] = 192
        c[c>255] = 255
        v_out[idx] = c
        return v_out.astype(np.uint8)

    

    @staticmethod
    def _char_to_float(v, p0, p25, p75, p100):
        v_in = np.fromstring(v, dtype=np.uint8)
        v_out = np.zeros(v_in.shape, dtype=float_cpu())
        idx = v_in <= 64
        v_out[idx] = p0 + (p25-p0)*v_in[idx]/64.0
        idx = np.logical_and(v_in>64, v_in<=192)
        v_out[idx] = p25 + (p75-p25)*(v_in[idx] - 64)/128.0
        idx = v_in > 192
        v_out[idx] = p75 + (p100-p75)*(v_in[idx] - 192)/63.0
        return v_out


    
    def to_ndarray(self):
        if self.data_format == 1:
            mat = np.zeros((self.num_rows, self.num_cols), dtype=float_cpu())
            header_offset = 20
            data_offset = header_offset+self.num_cols*8
            for i in xrange(self.num_cols):
                mat[:,i] = self._uncompress_column(
                    self.data[header_offset:header_offset+8],
                    self.data[data_offset:data_offset+self.num_rows])
                header_offset += 8
                data_offset += self.num_rows
        elif self.data_format == 2:
            #data = np.fromstring(self.data[20:], dtype=np.uint16)
            # mat = np.reshape(self.min_value + self.data_range/65535.0 * data, (self.num_rows, self.num_cols)).astype(float_cpu())
            mat = np.reshape(self._uint16_to_float(self.data[20:]),
                             (self.num_rows, self.num_cols)).astype(float_cpu(), copy=False)
        else:
            # data = np.fromstring(self.data[20:], dtype=np.uint8)
            # mat = np.reshape(self.min_value + self.data_range/255.0 * data, (self.num_rows, self.num_cols)).astype(float_cpu())
            mat = np.reshape(self._uint8_to_float(self.data[20:]),
                             (self.num_rows, self.num_cols)).astype(float_cpu(), copy=False)

        return mat


    
    def to_matrix(self):
        mat = self.to_ndarray()
        return KaldiMatrix(mat)
    

    
    @classmethod
    def read(cls, f, binary, row_offset=0, num_rows=0, sequential_mode=True):
        if binary:
            peekval = peek(f, binary)
            if peekval == b'C':
                token = read_token(f, binary)
                if token == 'CM':
                    data_format = 1
                elif token == 'CM2':
                    data_format = 2
                elif token == 'CM3':
                    data_format = 3
                else:
                    raise ValueError('Unexpected token %s' % token)
                
                header = struct.pack('<i', data_format)+f.read(16)
                header, num_cols, bytes_col_header, bytes_offset, bytes_col, bytes_left = cls._get_read_info(
                    header, row_offset, num_rows)
                
                if bytes_offset == 0 and bytes_left == 0:
                    data = header + f.read(bytes_col_header+num_cols*bytes_col)
                else:
                    col_header = f.read(bytes_col_header)
                    data = bytes()
                    for c in xrange(num_cols):
                        if bytes_offset > 0:
                            f.seek(bytes_offset, 1)
                        data += f.read(bytes_col)
                        if bytes_left > 0:
                            f.seek(bytes_left, 1)
                            
                    if bytes_col_header > 0:
                        data = header + col_header + data
                    else:
                        data = header + data
                    
                # header, bytes_col_header, bytes_offset, bytes_data, bytes_left = cls._get_read_info(
                #     header, row_offset, num_rows)
                
                # if bytes_offset == 0:
                #     data = header + f.read(bytes_col_header+bytes_data)
                # else:
                #     if bytes_col_header > 0:
                #         col_header = f.read(bytes_col_header)
                #         f.seek(bytes_offset, 1)
                #         data = header + col_header + f.read(bytes_data)
                #     else:
                #         f.seek(bytes_offset, 1)
                #         data = header + f.read(bytes_data)
                        
                # if bytes_left> 0 and sequential_mode:
                #     f.seek(bytes_left, 1)

                return cls(data)
            else:
                matrix = KaldiMatrix.read(f, binary, row_offset, num_rows)
                return cls.compress(matrix)
            
        assert row_offset == 0, 'row offset not supported in text mode because it is inefficient'
        matrix = KaldiMatrix.read(f, binary)
        return cls.compress(matrix)


    
    def write(self, f, binary):
        if binary:
            if self.data is not None:
                if self.data_format == 1:
                    write_token(f, binary, 'CM')
                elif self.data_format == 2:
                    write_token(f, binary, 'CM2')
                elif self.data_format == 3:
                    write_token(f, binary, 'CM3')
                f.write(self.data[4:])
            else:
                write_token(f, binary, 'CM')
                header = struct.pack('<ffii',0,0,0,0)
                f.write(header)
        else:
            self.to_matrix().write(f, binary)

                    
        
    @staticmethod
    def read_shape(f, binary, sequential_mode=True):
        if binary:
            peekval = peek(f, binary)
            if peekval == b'C':
                token = read_token(f, binary)
                if token == 'CM':
                    data_format = 1
                elif token == 'CM2':
                    data_format = 2
                elif token == 'CM3':
                    data_format = 3
                else:
                    raise ValueError('Unexpected token %s' % token)
                
                header = struct.pack('<i', data_format)+f.read(16)
                num_rows, num_cols = struct.unpack('<iffii', header)[-2:]
                if sequential_mode:
                    size = KaldiCompressedMatrix._data_size(header) - len(header)
                    f.seek(size, 1)

                return (num_rows, num_cols)
            else:
                matrix = KaldiMatrix.read(f, binary, row_offset, num_rows)
                return matrix.data.shape

        matrix = KaldiMatrix.read(f, binary)
        return matrix.data.shape



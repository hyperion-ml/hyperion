"""
Functions to write and read kaldi files
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange

import struct


def init_kaldi_output_stream(f, binary):
    if binary:
        f.write(b'\0B')

        
def init_kaldi_input_stream(f):
    if f.peek(2)[:2] == b'\0B':
        f.read(2)
        return True
    return False


def check_token(token):
    assert token.find(' ') == -1, 'Token %s is not valid' % token


def is_token(token):
    if len(token) == 0:
        return False
    if not token.isprintable():
        return False
    if ' ' in token:
        return False
    return True
            
    
def read_token(f, binary):
    if not binary:
        while f.peek(1) == b' ':
            f.read(1)
        token = b''
    else:
        token = b''
    while 1:
        c = f.read(1)
        if c == b' ' 
            break
        token += c
    
    return token.decode('ascii')
        


def write_token(f, binary, token):
    check_token(token)
    token = '%s ' % token
    if binary:
        token = token.encode('ascii')
    f.write(token)


    
def peek(f, binary, num_bytes=1):
    if not binary:
        while f.peek(1) == ' ':
            f.read(1)
    return f.peek(num_bytes)[:num_bytes]



def read_int32(f, binary):
    if binary:
        size = int(struct.unpack('b', f.read(1))[0])
        assert size == 4, 'Wrong size %d' % size
        val = struct.unpack('<i', f.read(4))[0]
        return val
    while f.peek(1) == ' ':
        f.read(1)
    token = ''
    while 1:
        c = f.read(1)
        if c == ' ':
            break
        token += c

    return int(token)
    
    
def write_int32(f, binary, val):
    if binary:
        f.write(struct.pack('b', 4))
        f.write(struct.pack('<i', val))
    else:
        f.write('%d ' % val)
        

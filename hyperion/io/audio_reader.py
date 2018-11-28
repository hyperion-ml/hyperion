"""
Class to read audio files
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import os
import logging
import subprocess
import soundfile as sf

import numpy as np

from ..hyp_defs import float_cpu
from ..utils.scp_list import SCPList

valid_ext = ['.wav', '.flac', '.ogg' , '.au', '.avr', '.caf', '.htk', '.iff', '.mat', '.mpc', '.oga', '.pvf', '.rf64', '.sd2', '.sds', '.sf', '.voc', 'w64', '.wve', '.xi']

class AudioReader(object):
    """Class to read audio files from wav, flac or pipe

       Attributes:
            file_path: scp file with formant file_key wavspecifier (audio_file/pipe).
            scp:       scp object
            scale:        Multiplies signal by scale factor
    """
    
    def __init__(self, file_path, scp=None, scale=2**15):
        self.file_path = file_path
        if scp is None:
            self.scp = SCPList.load(file_path, sep=' ')
        else:
            self.scp = scp
        self.scale = scale

        

    def __enter__(self):
        """Function required when entering contructions of type

           with AudioReader('file.h5') as f:
              keys, data = f.read()
        """
        return self


    
    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

           with DataReader('file.h5') as f:
              keys, data = f.read()
        """
        pass


    @staticmethod
    def read_wavspecifier(wavspecifier, scale=2**15):
        """Reads an audiospecifier (audio_file/pipe)
           It reads from pipe or from all the files that can be read 
           by `libsndfile <http://www.mega-nerd.com/libsndfile/#Features>`

        Args:
          wavspecifier: A pipe, wav, flac, ogg file etc.
          scale:        Multiplies signal by scale factor
        """
        wavspecifier = wavspecifier.strip()
        if wavspecifier[-1] == '|':
            return read_pipe(wavspecifier, scale)
        else:
            ext = os.path.splitext(wavspecifier)[1]
            if ext in valid_ext:
                x, fs = sf.read(wavspecifier, dtype=float_cpu())
                x *= scale
            else:
                raise Exception('Unknown format for %s' % (wavspecifier))

        return x, fs


    @staticmethod
    def read_pipe(wavspecifier, scale=2**15):
        """Reads wave file from a pipe
        Args:
          wavspecifier: Shell command with pipe output
          scale:        Multiplies signal by scale factor
        """
        proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDERR)
        pipe = proc.communicate()[0]
        x, fs = sf.read(io.BytesIO(pipe), dtype=float_cpu())
        x *= scale
        return x, fs
    

    def read(self):
        pass



class SequentialAudioReader(AudioReader):

    def __init__(self, file_path, scp=None, part_idx=1, num_parts=1):
        super(SequentialAudioReader, self).__init__(file_path, scp)
        self.cur_item = 0
        self.part_idx = 1
        self.num_parts = num_parts
        if self.num_parts > 1:
            self.scp = self.scp.split(self.part_idx, self.num_parts)


        
    def __iter__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialAudioReader(...)
        for key, s, fs in r:
           print(key)
           process(s)
        """
        return self


    
    def __next__(self):
        """Needed to build an iterator, e.g.:
        r = SequentialAudioReader(...)
        for key , s, fs in r:
           process(s)
        """
        key, x, fs = self.read(1)
        if len(key)==0:
            raise StopIteration
        return key[0], x[0], fs[0]
    

    
    def next(self):
        """__next__ for Python 2"""
        return self.__next__()


    def reset(self):
        """Returns the file pointer to the begining of the dataset, 
           then we can start reading the features again.
        """
        self.cur_item=0



    def eof(self):
        """End of file.

        Returns:
          True, when we have read all the recordings in the dataset.
        """
        return self.cur_item == len(self.scp)

    
    def read(self, num_records=0):
        """Reads next num_records audio files
        
        Args:
          num_records: Number of audio files to read.

        Returns:
          key: List of recording names.
          data: List of waveforms
        """
        if num_records == 0:
            num_records = len(self.scp) - self.cur_item

        keys = []
        data = []
        fs = []
        for i in xrange(num_records):
            if self.eof():
                break
            key, file_path, _, _ = self.scp[self.cur_item]
            x_i, fs_i = self.read_wavspecifier(file_path, self.scale)
            keys.append(key)
            data.append(x_i)
            fs.append(fs_i)
            self.cur_item += 1

        return keys, data, fs


    

class RandomAccessAudioReader(AudioReader):

    def __init__(self, file_path, scp=None):
        super(RandomAccessAudioReader, self).__init__(file_path, scp)
        


    def read(self, keys):
        """Reads the waveforms  for the recordings in keys.
        
        Args:
          keys: List of recording names.

        Returns:
          data: List of waveforms
        """
        if isinstance(keys, string_types):
            keys = [keys]

        data = []
        fs = []
        for i,key in enumerate(keys):
            if not (key in self.scp):
                raise Exception('Key %s not found' % key)

            file_path, _, _ = self.scp[key]
            x_i, fs_i = self.read_wavspecifier(file_path, self.scale)
            data.append(x_i)
            fs.append(fs_i)

        return data, fs


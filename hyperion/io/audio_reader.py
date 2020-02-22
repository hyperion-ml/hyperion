"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from six.moves import xrange
from six import string_types

import os
import logging
import io
import subprocess
import soundfile as sf

import numpy as np

from ..hyp_defs import float_cpu
from ..utils import SCPList, SegmentList

valid_ext = ['.wav', '.flac', '.ogg' , '.au', '.avr', '.caf', '.htk', '.iff', '.mat', '.mpc', '.oga', '.pvf', '.rf64', '.sd2', '.sds', '.sf', '.voc', 'w64', '.wve', '.xi']

class AudioReader(object):
    """Class to read audio files from wav, flac or pipe

       Attributes:
            file_path:     scp file with formant file_key wavspecifier (audio_file/pipe) or SCPList object.
            segments_path: segments file with format: segment_id file_id tbeg tend
            wav_scale:         multiplies signal by scale factor
    """
    
    def __init__(self, file_path, segments_path=None, wav_scale=2**15):
        self.file_path = file_path
        if isinstance(file_path, SCPList):
            self.scp = file_path
        else:
            self.scp = SCPList.load(file_path, sep=' ', is_wav=True)

        self.segments_path = segments_path
        if segments_path is None:
            self.segments = None
            self.with_segments = False
        else:
            self.with_segments = True
            if isinstance(file_path, SegmentList):
                self.segments = segments_path
            else:
                self.segments = SegmentList.load(segments_path, sep=' ', index_by_file=False)

        self.scale = wav_scale

        

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
            wavspecifier = wavspecifier[:-1]
            return AudioReader.read_pipe(wavspecifier, scale)
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
        # proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        proc = subprocess.Popen(wavspecifier, shell=True, stdout=subprocess.PIPE)
        pipe = proc.communicate()[0]
        if proc.returncode !=0:
            raise Exception('Wave read pipe command %s returned code %d' % (wavspecifier, proc.returncode))
        x, fs = sf.read(io.BytesIO(pipe), dtype=float_cpu())
        x *= scale
        return x, fs


    def _read_segment(self, segment):
        """Reads a wave segment

        Args:
          segment: pandas DataFrame (segment_id , file_id, tbeg, tend)
        Returns:
          Wave, sampling frequency
        """
        file_id = segment['file_id']
        t_beg = segment['tbeg']
        t_end = segment['tend']
        file_path, _, _ = self.scp[file_id]
        x_i, fs_i = self.read_wavspecifier(file_path, self.scale)
        num_samples_i = len(x_i)
        s_beg = int(t_beg * fs_i)
        if s_beg >= num_samples_i:
            raise Exception('segment %s tbeg=%.2f (num_sample=%d) longer that wav file %s (num_samples=%d)' % (
                key, tbeg, sbeg, file_id, num_samples_i))

        s_end = int(t_end * fs_i)
        if s_end > num_samples_i or t_end < 0:
            s_end = num_samples_i
                    
        x_i = x_i[s_beg:s_end]
        return x_i, fs_i

    

    def read(self):
        pass



class SequentialAudioReader(AudioReader):

    def __init__(self, file_path, segments_path=None, wav_scale=2**15, part_idx=1, num_parts=1):
        super(SequentialAudioReader, self).__init__(file_path, segments_path, wav_scale=wav_scale)
        self.cur_item = 0
        self.part_idx = part_idx
        self.num_parts = num_parts
        if self.num_parts > 1:
            if self.with_segments:
                self.segments = self.segments.split(self.part_idx, self.num_parts)
            else:
                self.scp = self.scp.split(self.part_idx, self.num_parts, group_by_key=False)


        
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
        if self.with_segments:
            return self.cur_item == len(self.segments)
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
            if self.with_segments:
                segment = self.segments[self.cur_item]
                key = segment['segment_id']
                x_i, fs_i = self._read_segment(segment)
            else:
                key, file_path, _, _ = self.scp[self.cur_item]
                x_i, fs_i = self.read_wavspecifier(file_path, self.scale)

            keys.append(key)
            data.append(x_i)
            fs.append(fs_i)
            self.cur_item += 1

        return keys, data, fs

    
    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('part_idx', 'num_parts','wav_scale')
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
            
        # parser.add_argument(p1+'scp-sep', dest=(p2+'scp_sep'), default=' ',
        #                     help=('scp file field separator'))
        parser.add_argument(p1+'wav-scale', dest=(p2+'wav_scale'), default=2**15, type=float,
                             help=('multiplicative factor for waveform'))
        parser.add_argument(p1+'part-idx', dest=(p2+'part_idx'), type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))
        parser.add_argument(p1+'num-parts', dest=(p2+'num_parts'), type=int, default=1,
                            help=('splits the list of files in num-parts and process part_idx'))

    

class RandomAccessAudioReader(AudioReader):

    def __init__(self, file_path, segments_path=None, wav_scale=2**15):
        super(RandomAccessAudioReader, self).__init__(file_path, segments_path, wav_scale)



    def read(self, keys):
        """Reads the waveforms  for the recordings in keys.
        
        Args:
          keys: List of recording/segment_ids names.

        Returns:
          data: List of waveforms
        """
        if isinstance(keys, string_types):
            keys = [keys]

        data = []
        fs = []
        for i,key in enumerate(keys):
            if self.with_segments:
                if not (key in self.segments):
                    raise Exception('Key %s not found' % key)
                
                segment = self.segments[key]
                x_i, fs_i = self._read_segment(segment)
            else:
                if not (key in self.scp):
                    raise Exception('Key %s not found' % key)

                file_path, _, _ = self.scp[key]
                x_i, fs_i = self.read_wavspecifier(file_path, self.scale)

            data.append(x_i)
            fs.append(fs_i)

        return data, fs


    @staticmethod
    def filter_args(prefix=None, **kwargs):
        if prefix is None:
            p = ''
        else:
            p = prefix + '_'
        valid_args = ('wav_scale')
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
            
        parser.add_argument(p1+'wav-scale', dest=(p2+'wav_scale'), default=2**15, type=float,
                             help=('multiplicative factor for waveform'))

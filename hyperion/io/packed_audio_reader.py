"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import soundfile as sf

import time
import math
import logging
import numpy as np
import multiprocessing
from copy import deepcopy

from ..hyp_defs import float_cpu
from ..utils import SCPList, SegmentList


class PackedAudioReader(object):
    """Base class to read audio utterances which have been packed in few larger audio files (wav, flac, ogg)

    Attributes:
         file_path: scp file with formant utterance_key packed_audio_file_path:offset[first_sample:last_sample]
                    where:
                             offset = number of samples wrt the begining of the file where utterance_key is found
                             first_sample = first sample to read wrt the offset
                             last_sample = last sample to read wrt the offset
         segments_path: Kaldi segments file with format: segment_id file_id tbeg tend (optional)
         wav_scale: multiplies signal by scale factor typically 2**15-1 to transform from (-1,1] to 16 bits
                    dynamic range
    """

    def __init__(self, file_path, segments_path=None, wav_scale=2 ** 15 - 1):
        self.file_path = file_path
        if isinstance(file_path, SCPList):
            self.scp = file_path
        else:
            self.scp = SCPList.load(file_path, sep=" ")

        self.segments_path = segments_path
        if segments_path is None:
            self.segments = None
            self.with_segments = False
        else:
            self.with_segments = True
            if isinstance(file_path, SegmentList):
                self.segments = segments_path
            else:
                self.segments = SegmentList.load(
                    segments_path, sep=" ", index_by_file=False
                )

        self.scale = wav_scale

    def __enter__(self):
        """Function required when entering contructions of type

        with PackedAudioReader('file.scp') as f:
           keys, data = f.read()
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Function required when exiting from contructions of type

        with PackedAudioReader('file.scp') as f:
           keys, data = f.read()
        """
        pass

    @staticmethod
    def _combine_ranges(read_range, offset, num_samples):
        """Combines two sample ranges.
           One is the range in the scp file, e.g, in the scp file
              recording1  file1.ark:34[3:40]
              recording2  file1.ark:100[5:20]

              [3:40] and [5:20] are frame ranges.

           The user can decide to just read a subset of that, e.g.,
           read 10 samples starting at sample 1.
           If we combine that with the range [3:40], the function returns.
           offset=4 (3+1) and num_samples=10.

        Args:
          read_range: sample range from scp file. It is a tuple with the
             first sample and number of samples to read.
          offset: User defined offset samples
          num_samples: User defined number of samples to read, it it is 0, we read
                    all the rows defined in the scp read_range.

        Returns:
          Combined offset, first sample of the recording to read.
          Combined number of samples to read.
        """
        if read_range is None:
            return offset, num_samples

        if num_samples == 0:
            num_samples = read_range[1] - offset
            assert num_samples > 0
        else:
            assert read_range[1] - offset >= num_samples

        offset = offset + read_range[0]
        return offset, num_samples


class SequentialPackedAudioReader(PackedAudioReader):
    """Class to read sequentially audio utterances which have been packed in few larger audio files (wav, flac, ogg)

    Attributes:
         file_path: scp file with formant utterance_key packed_audio_file_path:offset[first_sample:last_sample]
                    where:
                             offset = number of samples wrt the begining of the file where utterance_key is found
                             first_sample = first sample to read wrt the offset
                             last_sample = last sample to read wrt the offset
         segments_path: Kaldi segments file with format: segment_id file_id tbeg tend (optional)
         wav_scale: multiplies signal by scale factor typically 2**15-1 to transform from (-1,1] to 16 bits
                    dynamic range
         part_idx: split scp file into num_parts and uses part part_idx
         num_parts: number of parts to split the scp file
    """

    def __init__(
        self,
        file_path,
        segments_path=None,
        wav_scale=2 ** 15 - 1,
        part_idx=1,
        num_parts=1,
    ):
        super().__init__(file_path, segments_path, wav_scale=wav_scale)
        self.cur_item = 0
        self.part_idx = part_idx
        self.num_parts = num_parts
        self.f = None
        self.lock = multiprocessing.Lock()
        self.cur_file = None
        if self.num_parts > 1:
            if self.with_segments:
                self.segments = self.segments.split(self.part_idx, self.num_parts)
            else:
                self.scp = self.scp.split(
                    self.part_idx, self.num_parts, group_by_key=False
                )

    def close(self):
        """Closes input file."""
        if self.f is not None:
            self.f.close()
            self.f = None

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
        if len(key) == 0:
            raise StopIteration
        return key[0], x[0], fs[0]

    def next(self):
        """__next__ for Python 2"""
        return self.__next__()

    def reset(self):
        """Returns the file pointer to the begining of the dataset,
        then we can start reading the features again.
        """
        self.close()
        self.cur_item = 0

    def eof(self):
        """End of file.

        Returns:
          True, when we have read all the recordings in the dataset.
        """
        if self.with_segments:
            return self.cur_item == len(self.segments)
        return self.cur_item == len(self.scp)

    def _open_archive(self, file_path, offset=None):
        """Opens the current file if it is not open and moves the
           file pointer to a given position.
           Closes previous open audio files.

        Args:
          file_path: File from which we want to read the next feature matrix.
          offset: sample position where utterance starts in the packed audio file.
        """
        if self.f is None or file_path != self.cur_file:
            self.close()
            self.cur_file = file_path
            self.f = sf.SoundFile(file_path, "r")

        if offset is not None:
            self.f.seek(offset)

    def read_num_samples(self, num_records=0):
        """Reads the number of samples in the utterances of the packed audio file

        Args:
          num_records: How many utterances to read, if num_records=0 it
                       reads all utterances

        Returns:
          List of num_records recording names.
          Integer numpy array with num_records num samples
        """
        if num_records == 0:
            if self.with_segments:
                num_records = len(self.segments) - self.cur_item
            else:
                num_records = len(self.scp) - self.cur_item

        keys = []
        num_samples = np.zeros((num_records,), dtype=np.int)
        for i in range(num_records):
            if self.eof():
                num_samples = num_samples[:i]
                break

            if self.with_segments:
                segment = self.segments[self.cur_item]
                key_i = segment["segment_id"]
                file_id = segment["file_id"]
                t_beg = segment["tbeg"]
                t_end = segment["tend"]
                file_path, _, range_spec = self.scp[file_id]
                self._open_archive(file_path)
                fs = self.f.samplerate
                num_samples_i = int(math.floor((t_end - t_beg) * fs))
                max_samples = range_spec[1]
                if num_samples_i > max_samples:
                    logging.warning(
                        "Duration of segment %s in segments-file (%d samples) > "
                        "full utterance %s duration (%d)"
                        % (segment["segment_id"], num_samples_i, file_id, max_samples)
                    )
                    num_samples_i = max_samples
            else:
                key_i, file_path, offset, range_spec = self.scp[self.cur_item]
                num_samples_i = range_spec[1]

            keys.append(key_i)
            num_samples[i] = num_samples_i
            self.cur_item += 1

        return keys, num_samples

    def read_time_duration(self, num_records=0):
        """Reads the duration in secs. of the utterances of the packed audio file

        Args:
          num_records: How many utterances to read, if num_records=0 it
                       reads all utterances

        Returns:
          List of num_records recording names.
          Float numpy array with num_records lengths in secs
        """
        if num_records == 0:
            if self.with_segments:
                num_records = len(self.segments) - self.cur_item
            else:
                num_records = len(self.scp) - self.cur_item

        keys = []
        time_dur = np.zeros((num_records,), dtype=np.float)
        for i in range(num_records):
            if self.eof():
                time_dur = time_dur[:i]
                break

            if self.with_segments:
                segment = self.segments[self.cur_item]
                key_i = segment["segment_id"]
                t_beg = segment["tbeg"]
                t_end = segment["tend"]
                time_dur_i = t_end - t_beg
            else:
                key_i, file_path, _, range_spec = self.scp[self.cur_item]
                self._open_archive(file_path)
                fs = self.f.samplerate
                time_dur_i = range_spec[1] / fs

            keys.append(key_i)
            time_dur[i] = time_dur_i
            self.cur_item += 1

        return keys, time_dur

    def read(self, num_records=0, time_offset=0, time_durs=0):
        """Reads next num_records audio files

        Args:
          num_records: Number of audio files to read.
          time_offset: List of floats indicating the start time to read in the utterance.
          time_durs: List of floats indicating the number of seconds to read from each utterance
        Returns:
          key: List of recording names.
          data: List of waveforms
        """
        if num_records == 0:
            if self.with_segments:
                num_records = len(self.segments) - self.cur_item
            else:
                num_records = len(self.scp) - self.cur_item

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        keys = []
        data = []
        fs = []
        with self.lock:
            for i in range(num_records):
                if self.eof():
                    break

                offset_i = time_offset[i] if offset_is_list else time_offset
                dur_i = time_durs[i] if dur_is_list else time_durs

                if self.with_segments:
                    segment = self.segments[self.cur_item]
                    key = segment["segment_id"]

                    segment_range_spec = (
                        segment["tbeg"],
                        segment["tend"] - segment["tbeg"],
                    )
                    offset_i, dur_i = self._combine_ranges(
                        segment_range_spec, offset_i, dur_i
                    )
                    file_path, offset, range_spec = self.scp[segment["file_id"]]
                else:
                    key, file_path, offset, range_spec = self.scp[self.cur_item]

                self._open_archive(file_path)
                fs_i = self.f.samplerate
                offset_i = int(math.floor(offset_i * fs_i))
                dur_i = int(math.floor(dur_i * fs_i))
                offset_i, dur_i = self._combine_ranges(range_spec, offset_i, dur_i)

                self.f.seek(offset + offset_i)
                x_i = self.scale * self.f.read(dur_i, dtype=float_cpu())

                keys.append(key)
                data.append(x_i)
                fs.append(fs_i)
                self.cur_item += 1

        return keys, data, fs

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("part_idx", "num_parts", "wav_scale")
        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "wav-scale",
            default=2 ** 15 - 1,
            type=float,
            help=("multiplicative factor for waveform"),
        )
        try:
            parser.add_argument(
                p1 + "part-idx",
                type=int,
                default=1,
                help=("splits the list of files in num-parts and " "process part_idx"),
            )
            parser.add_argument(
                p1 + "num-parts",
                type=int,
                default=1,
                help=("splits the list of files in num-parts and " "process part_idx"),
            )
        except:
            pass


class RandomAccessPackedAudioReader(PackedAudioReader):
    def __init__(self, file_path, segments_path=None, wav_scale=2 ** 15 - 1):
        super().__init__(file_path, segments_path, wav_scale)

        archives, archive_idx = np.unique(self.scp.file_path, return_inverse=True)
        self.archives = archives
        self.archive_idx = archive_idx
        self.f = [None] * len(self.archives)
        self.locks = [multiprocessing.Lock() for i in range(len(self.archives))]

    def close(self):
        """Closes all the open audio files."""
        for f in self.f:
            if f is not None:
                f.close()
        self.f = [None] * len(self.f)

    def _open_archive(self, key_idx, offset=None):
        """Opens the packed audio file correspoding to a given utterance
           if it is not already open and moves the file pointer to the
           point where we can read the utterance

           If the file was already open, it only moves the file pointer.

        Args:
          key_idx: Integer position of the utterance in the scp file.
          offset: sample where the utterance starts in the packed audio  file.

        Returns:
          soundfile object
          lock object correcponding to the soundfile object
        """
        archive_idx = self.archive_idx[key_idx]
        with self.locks[archive_idx]:
            if self.f[archive_idx] is None:
                self.f[archive_idx] = sf.SoundFile(self.archives[archive_idx], "r")

            f = self.f[archive_idx]
            if offset is not None:
                f.seek(offset)

        return f, self.locks[archive_idx]

    def read_num_samples(self, keys):
        """Reads the number of samples in the utterances of the packed audio file

        Args:
          keys: List of recording/segment_ids names.

        Returns:
          Integer numpy array with num_records num samples
        """
        if isinstance(keys, str):
            keys = [keys]

        num_samples = np.zeros((len(keys),), dtype=np.int)
        for i, key in enumerate(keys):

            if self.with_segments:
                if not (key in self.segments):
                    raise Exception("Key %s not found" % key)

                segment = self.segments[key]
                file_id = segment["file_id"]
                t_beg = segment["tbeg"]
                t_end = segment["tend"]
                index = self.scp.get_index(segment["file_id"])
                _, file_path, offset, range_spec = self.scp[index]
                f, lock = self._open_archive(index)
                fs = f.samplerate
                num_samples_i = int(math.floor((t_end - t_beg) * fs))
                max_samples = range_spec[1]
                if num_samples_i > max_samples:
                    logging.warning(
                        "Duration of segment %s in segments-file (%d samples) > "
                        "full utterance %s duration (%d)"
                        % (segment["segment_id"], num_samples_i, file_id, max_samples)
                    )
                    num_samples_i = max_samples
            else:
                if not (key in self.scp):
                    raise Exception("Key %s not found" % key)

                file_path, offset, range_spec = self.scp[key]
                num_samples_i = range_spec[1]

            num_samples[i] = num_samples_i

        return num_samples

    def read_time_duration(self, keys):
        """Reads the duration in secs. of the utterances of the packed audio file

        Args:
          keys: List of recording/segment_ids names.

        Returns:
          Float numpy array with num_records lengths in secs
        """
        if isinstance(keys, str):
            keys = [keys]

        time_dur = np.zeros((len(keys),), dtype=np.float)
        for i, key in enumerate(keys):

            if self.with_segments:
                if not (key in self.segments):
                    raise Exception("Key %s not found" % key)

                segment = self.segments[key]
                t_beg = segment["tbeg"]
                t_end = segment["tend"]
                time_dur_i = t_end - t_beg
            else:
                if not (key in self.scp):
                    raise Exception("Key %s not found" % key)
                index = self.scp.get_index(key)
                _, file_path, offset, range_spec = self.scp[index]
                f, lock = self._open_archive(index)
                fs = f.samplerate
                time_dur_i = range_spec[1] / fs

            time_dur[i] = time_dur_i

        return time_dur

    def read(self, keys, time_offset=0, time_durs=0):
        """Reads the waveforms  for the recordings in keys.

        Args:
          keys: List of recording/segment_ids names.
          time_offset: List of floats indicating the start time to read in the utterance.
          time_durs: List of floats indicating the number of seconds to read from each utterance

        Returns:
          data: List of waveforms
        """
        if isinstance(keys, str):
            keys = [keys]

        offset_is_list = isinstance(time_offset, (list, np.ndarray))
        dur_is_list = isinstance(time_durs, (list, np.ndarray))

        data = []
        fs = []
        for i, key in enumerate(keys):

            offset_i = time_offset[i] if offset_is_list else time_offset
            dur_i = time_durs[i] if dur_is_list else time_durs
            # t1= time.time()
            if self.with_segments:
                if not (key in self.segments):
                    raise Exception("Key %s not found" % key)

                segment = self.segments[key]
                segment_range_spec = (
                    segment["tbeg"],
                    segment["tend"] - segment["tbeg"],
                )
                offset_i, dur_i = self._combine_ranges(
                    segment_range_spec, offset_i, dur_i
                )
                index = self.scp.get_index(segment["file_id"])
                _, file_path, offset, range_spec = self.scp[index]
            else:
                if not (key in self.scp):
                    raise Exception("Key %s not found" % key)

                index = self.scp.get_index(key)
                _, file_path, offset, range_spec = self.scp[index]
            # t2=time.time()
            # aid = self.archive_idx[index]
            f, lock = self._open_archive(index)
            # while lock.locked():
            #     logging.info('checking locked {} {} {} {} {} {} {} {}'.format(
            #         index, aid, lock, lock.locked(),  key, offset, offset_i, dur_i))

            # l=True
            # logging.info('checking unlocked {} {} {} {} {} {} {} {}'.format(
            #     index, aid, lock, l,  key, offset, offset_i, dur_i))
            with lock:
                # t3 = time.time()
                # logging.info('lock {}'.format(aid))
                fs_i = f.samplerate
                offset_i = int(math.floor(offset_i * fs_i))
                dur_i = int(math.floor(dur_i * fs_i))
                offset_i, dur_i = self._combine_ranges(range_spec, offset_i, dur_i)
                # t4=time.time()
                cur_pos = f.tell()
                f.seek((offset + offset_i - cur_pos), sf.SEEK_CUR)
                # t5=time.time()
                x_i = self.scale * f.read(dur_i, dtype=float_cpu())
                # t6=time.time()
                # logging.info('time={} {} {} {} {} {}'.format(t6-t1,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5))
                # try:
                #     logging.info('par {} {} {} {} {} {} {} {}'.format(
                #         index, aid, lock, l,  key, offset, offset_i, dur_i))
                #     f.seek(offset+offset_i)
                #     x_i = self.scale * f.read(dur_i, dtype=float_cpu())
                # except Exception:
                #     try:
                #         logging.info('except par {} {} {} {} {} {} {} {}'.format(
                #             index, aid, lock, l,  key, offset, offset_i, dur_i))
                #         f.seek(0)
                #         f.seek(offset+offset_i)
                #         x_i = self.scale * f.read(dur_i, dtype=float_cpu())
                #     except Exception as e:
                #         logging.info('except2 par {} {} {} {} {} {} {} {}'.format(
                #             index, aid, lock, l,  key, offset, offset_i, dur_i))
                #         #time.sleep(10)
                #         raise e

                # logging.info('unlock {}'.format(aid))

            data.append(x_i)
            fs.append(fs_i)

        return data, fs

    @staticmethod
    def filter_args(**kwargs):
        valid_args = ("wav_scale",)

        return dict((k, kwargs[k]) for k in valid_args if k in kwargs)

    @staticmethod
    def add_class_args(parser, prefix=None):
        if prefix is None:
            p1 = "--"
        else:
            p1 = "--" + prefix + "."

        parser.add_argument(
            p1 + "wav-scale",
            default=2 ** 15,
            type=float,
            help=("multiplicative factor for waveform"),
        )

    add_argparse_args = add_class_args

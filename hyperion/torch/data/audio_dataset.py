"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
from jsonargparse import ActionYesNo, ArgumentParser, ActionParser
import time
import math

import numpy as np
import pandas as pd

import torch
import torchaudio.transforms as tat

from ..torch_defs import floatstr_torch
from ...io import RandomAccessAudioReader as AR

# from ...utils.utt2info import Utt2Info
from ...np.augment import SpeechAugment


import k2
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
import torch.distributed as dist

from hyperion.np import augment

# class AudioDataset1(Dataset):
#     def __init__(
#         self,
#         audio_file,
#         key_file,
#         class_file=None,
#         time_durs_file=None,
#         min_chunk_length=1,
#         max_chunk_length=None,
#         aug_cfg=None,
#         return_fullseqs=False,
#         return_class=True,
#         return_clean_aug_pair=False,
#         transpose_input=False,
#         wav_scale=2 ** 15 - 1,
#         is_val=False,
#     ):

#         try:
#             rank = dist.get_rank()
#             world_size = dist.get_world_size()
#         except:
#             rank = 0
#             world_size = 1

#         self.rank = rank
#         self.world_size = world_size

#         if rank == 0:
#             logging.info("opening dataset %s", audio_file)
#         self.r = AR(audio_file, wav_scale=wav_scale)
#         if rank == 0:
#             logging.info("loading utt2info file %s" % key_file)
#         self.u2c = Utt2Info.load(key_file, sep=" ")
#         if rank == 0:
#             logging.info("dataset contains %d seqs" % self.num_seqs)

#         self.is_val = is_val
#         self._read_time_durs_file(time_durs_file)

#         self._prune_short_seqs(min_chunk_length)

#         self.short_seq_exist = self._seq_shorter_than_max_length_exists(
#             max_chunk_length
#         )

#         self._prepare_class_info(class_file)

#         if max_chunk_length is None:
#             max_chunk_length = min_chunk_length
#         self._min_chunk_length = min_chunk_length
#         self._max_chunk_length = max_chunk_length

#         self.return_fullseqs = return_fullseqs
#         self.return_class = return_class
#         self.return_clean_aug_pair = return_clean_aug_pair

#         self.transpose_input = transpose_input

#         self.augmenter = None
#         self.reverb_context = 0
#         if aug_cfg is not None:
#             self.augmenter = SpeechAugment.create(
#                 aug_cfg, random_seed=112358 + 1000 * rank
#             )
#             self.reverb_context = self.augmenter.max_reverb_context

#     def _read_time_durs_file(self, file_path):
#         if self.rank == 0:
#             logging.info("reading time_durs file %s" % file_path)
#         nf_df = pd.read_csv(file_path, header=None, sep=" ")
#         nf_df.index = nf_df[0]
#         self._seq_lengths = nf_df.loc[self.u2c.key, 1].values

#     @property
#     def wav_scale(self):
#         return self.r.wav_scale

#     @property
#     def num_seqs(self):
#         return len(self.u2c)

#     def __len__(self):
#         return self.num_seqs

#     @property
#     def seq_lengths(self):
#         return self._seq_lengths

#     @property
#     def total_length(self):
#         return np.sum(self.seq_lengths)

#     @property
#     def min_chunk_length(self):
#         if self.return_fullseqs:
#             self._min_chunk_length = np.min(self.seq_lengths)
#         return self._min_chunk_length

#     @property
#     def max_chunk_length(self):
#         if self._max_chunk_length is None:
#             self._max_chunk_length = np.max(self.seq_lengths)
#         return self._max_chunk_length

#     @property
#     def min_seq_length(self):
#         return np.min(self.seq_lengths)

#     @property
#     def max_seq_length(self):
#         return np.max(self.seq_lengths)

#     def _prune_short_seqs(self, min_length):
#         if self.rank == 0:
#             logging.info("pruning short seqs")
#         keep_idx = self.seq_lengths >= min_length
#         self.u2c = self.u2c.filter_index(keep_idx)
#         self._seq_lengths = self.seq_lengths[keep_idx]
#         if self.rank == 0:
#             logging.info(
#                 "pruned seqs with min_length < %f,"
#                 "keep %d/%d seqs" % (min_length, self.num_seqs, len(keep_idx))
#             )

#     def _prepare_class_info(self, class_file):
#         class_weights = None
#         if class_file is None:
#             classes, class_idx = np.unique(self.u2c.info, return_inverse=True)
#             class2idx = {k: i for i, k in enumerate(classes)}
#         else:
#             if self.rank == 0:
#                 logging.info("reading class-file %s" % (class_file))
#             class_info = pd.read_csv(class_file, header=None, sep=" ")
#             class2idx = {str(k): i for i, k in enumerate(class_info[0])}
#             class_idx = np.array([class2idx[k] for k in self.u2c.info], dtype=int)
#             if class_info.shape[1] == 2:
#                 class_weights = np.array(class_info[1]).astype(
#                     floatstr_torch(), copy=False
#                 )

#         self.num_classes = len(class2idx)

#         class2utt_idx = {}
#         class2num_utt = np.zeros((self.num_classes,), dtype=int)

#         for k in range(self.num_classes):
#             idx = (class_idx == k).nonzero()[0]
#             class2utt_idx[k] = idx
#             class2num_utt[k] = len(idx)
#             if class2num_utt[k] == 0:
#                 if not self.is_val:
#                     logging.warning("class %d doesn't have any samples" % (k))
#                 if class_weights is None:
#                     class_weights = np.ones((self.num_classes,), dtype=floatstr_torch())
#                 class_weights[k] = 0

#         count_empty = np.sum(class2num_utt == 0)
#         if count_empty > 0:
#             logging.warning("%d classes have 0 samples" % (count_empty))

#         self.utt_idx2class = class_idx
#         self.class2utt_idx = class2utt_idx
#         self.class2num_utt = class2num_utt
#         if class_weights is not None:
#             class_weights /= np.sum(class_weights)
#             class_weights = torch.Tensor(class_weights)
#         self.class_weights = class_weights

#         if self.short_seq_exist:
#             # if there are seq shorter than max_chunk_lenght we need some extra variables
#             # we will need class_weights to put to 0 classes that have all utts shorter than the batch chunk length
#             if self.class_weights is None:
#                 self.class_weights = torch.ones((self.num_classes,))

#             # we need the max length of the utterances of each class
#             class2max_length = torch.zeros((self.num_classes,), dtype=torch.float)
#             for c in range(self.num_classes):
#                 if class2num_utt[c] > 0:
#                     class2max_length[c] = np.max(
#                         self.seq_lengths[self.class2utt_idx[c]]
#                     )

#             self.class2max_length = class2max_length

#     def _seq_shorter_than_max_length_exists(self, max_length):
#         return np.any(self.seq_lengths < max_length)

#     @property
#     def var_chunk_length(self):
#         return self.min_chunk_length < self.max_chunk_length

#     def get_random_chunk_length(self):

#         if self.var_chunk_length:
#             return (
#                 torch.rand(size=(1,)).item()
#                 * (self.max_chunk_length - self.min_chunk_length)
#                 + self.min_chunk_length
#             )

#         return self.max_chunk_length

#     def __getitem__(self, index):
#         # logging.info('{} {} {} get item {}'.format(
#         #     self, os.getpid(), threading.get_ident(), index))
#         if self.return_fullseqs:
#             return self._get_fullseq(index)
#         else:
#             return self._get_random_chunk(index)

#     def _get_fullseq(self, index):
#         key = self.u2c.key[index]
#         x, fs = self.r.read([key])
#         x = x[0].astype(floatstr_torch(), copy=False)
#         x_clean = x
#         if self.augmenter is not None:
#             x, aug_info = self.augmenter(x)

#         if self.transpose_input:
#             x = x[None, :]
#             if self.return_clean_aug_pair:
#                 x_clean = x_clean[None, :]

#         if self.return_clean_aug_pair:
#             r = x, x_clean

#         if not self.return_class:
#             return r

#         class_idx = self.utt_idx2class[index]
#         r = *r, class_idx
#         return r

#     def _get_random_chunk(self, index):

#         if len(index) == 2:
#             index, chunk_length = index
#         else:
#             chunk_length = self.max_chunk_length

#         key = self.u2c.key[index]

#         full_seq_length = self.seq_lengths[index]
#         assert (
#             chunk_length <= full_seq_length
#         ), "chunk_length(%d) <= full_seq_length(%d)" % (chunk_length, full_seq_length)

#         time_offset = torch.rand(size=(1,)).item() * (full_seq_length - chunk_length)
#         reverb_context = min(self.reverb_context, time_offset)
#         time_offset -= reverb_context
#         read_chunk_length = chunk_length + reverb_context

#         # logging.info('get-random-chunk {} {} {} {} {}'.format(index, key, time_offset, chunk_length, full_seq_length ))
#         x, fs = self.r.read([key], time_offset=time_offset, time_durs=read_chunk_length)

#         # try:
#         #     x, fs = self.r.read([key], time_offset=time_offset,
#         #                     time_durs=read_chunk_length)
#         # except:
#         #     # some files produce error in the fseek after reading the data,
#         #     # this seems an issue from pysoundfile or soundfile lib itself
#         #     # reading from a sligthly different starting position seems to solve the problem in most cases
#         #     try:
#         #         logging.info('error-1 reading at key={} totol_dur={} offset={} read_chunk_length={}, retrying...'.format(
#         #             key, full_seq_length, time_offset, read_chunk_length))
#         #         time_offset = math.floor(time_offset)
#         #         x, fs = self.r.read([key], time_offset=time_offset,
#         #                             time_durs=read_chunk_length)
#         #     except:
#         #         try:
#         #             # if changing the value of time-offset doesn't solve the issue, we try to read from
#         #             # from time-offset to the end of the file, and remove the extra frames later
#         #             logging.info('error-2 reading at key={} totol_dur={} offset={} retrying reading until end-of-file ...'.format(
#         #                 key, full_seq_length, time_offset))
#         #             x, fs = self.r.read([key], time_offset=time_offset)
#         #             x = [x[0][:int(read_chunk_length * fs[0])]]
#         #         except:
#         #             # try to read the full file
#         #             logging.info('error-3 reading at key={} totol_dur={} retrying reading full file ...'.format(
#         #                 key, full_seq_length))
#         #             x, fs = self.r.read([key])
#         #             x = [x[0][:int(read_chunk_length * fs[0])]]

#         x = x[0]
#         fs = fs[0]

#         x_clean = x
#         logging.info("hola1")
#         if self.augmenter is not None:
#             logging.info("hola2")
#             chunk_length_samples = int(chunk_length * fs)
#             end_idx = len(x)
#             reverb_context_samples = end_idx - chunk_length_samples
#             assert reverb_context_samples >= 0, (
#                 "key={} time-offset={}, read-chunk={} "
#                 "read-x-samples={}, chunk_samples={}, reverb_context_samples={}"
#             ).format(
#                 key,
#                 time_offset,
#                 read_chunk_length,
#                 end_idx,
#                 chunk_length_samples,
#                 reverb_context_samples,
#             )
#             # end_idx = reverb_context_samples + chunk_length_samples
#             x, aug_info = self.augmenter(x)
#             x = x[reverb_context_samples:end_idx]
#             if self.return_clean_aug_pair:
#                 x_clean = x_clean[reverb_context_samples:end_idx]
#                 x_clean = x_clean.astype(floatstr_torch(), copy=False)
#             # x_clean = x_clean[reverb_context_samples:]
#             # logging.info('augmentation x-clean={}, x={}, aug_info={}'.format(
#             #    x_clean.shape, x.shape, aug_info))
#         #     if len(x) != 64000:
#         #         logging.info('x!=4s, {} {} {} {} {} {} {} {}'.format(len(x),reverb_context, reverb_context_samples, chunk_length, chunk_length_samples, end_idx, fs, read_chunk_length))

#         # if len(x) != 64000:
#         #         logging.info('x!=4s-2, {} {} {} {}'.format(len(x), chunk_length, fs, read_chunk_length))

#         if self.transpose_input:
#             x = x[None, :]
#             if self.return_clean_aug_pair:
#                 x_clean = x_clean[None, :]

#         x = x.astype(floatstr_torch(), copy=False)
#         if self.return_clean_aug_pair:
#             r = x, x_clean
#         else:
#             r = (x,)

#         if not self.return_class:
#             return r

#         class_idx = self.utt_idx2class[index]
#         r = *r, class_idx
#         return r

#     @staticmethod
#     def filter_args(**kwargs):

#         ar_args = AR.filter_args(**kwargs)
#         valid_args = (
#             "audio_file",
#             "key_file",
#             "aug_cfg",
#             "path_prefix",
#             "class_file",
#             "time_durs_file",
#             "min_chunk_length",
#             "max_chunk_length",
#             "return_fullseqs",
#             "part_idx",
#             "num_parts",
#         )
#         args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
#         args.update(ar_args)
#         return args

#     @staticmethod
#     def add_class_args(parser, prefix=None, skip={"audio_file", "key_file"}):
#         if prefix is not None:
#             outer_parser = parser
#             parser = ArgumentParser(prog="")

#         if "audio_file" not in skip:
#             parser.add_argument(
#                 "--audio-file",
#                 required=True,
#                 help=("audio manifest file"),
#             )

#         if "key_file" not in skip:
#             parser.add_argument(
#                 "--key-file",
#                 required=True,
#                 help=("key manifest file"),
#             )

#         parser.add_argument(
#             "--class-file",
#             default=None,
#             help=("ordered list of classes keys, it can contain class weights"),
#         )

#         parser.add_argument(
#             "--time-durs-file", default=None, help=("utt to duration in secs file")
#         )

#         parser.add_argument(
#             "--aug-cfg",
#             default=None,
#             help=("augmentation configuration file."),
#         )

#         parser.add_argument(
#             "--min-chunk-length",
#             type=float,
#             default=None,
#             help=("minimum length of sequence chunks"),
#         )
#         parser.add_argument(
#             "--max-chunk-length",
#             type=float,
#             default=None,
#             help=("maximum length of sequence chunks"),
#         )

#         parser.add_argument(
#             "--return-fullseqs",
#             default=False,
#             action="store_true",
#             help=("returns full sequences instead of chunks"),
#         )

#         AR.add_class_args(parser)
#         if prefix is not None:
#             outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
#             # help='audio dataset options')

#     add_argparse_args = add_class_args


from ...utils.class_info import ClassInfo
from ...utils.segment_set import SegmentSet
from ...utils.text import read_text

class AudioDataset(Dataset):
    def __init__(
        self,
        audio_file,
        segments_file,
        class_names=None,
        class_files=None,
        bpe_model=None,
        text_file=None,
        time_durs_file=None,
        aug_cfgs=None,
        num_augs=1,
        return_segment_info=None,
        return_orig=False,
        target_sample_freq=None,
        wav_scale=2 ** 15 - 1,
        is_val=False,
    ):

        super().__init__()
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        if rank == 0:
            logging.info("opening audio reader %s", audio_file)

        self.r = AR(audio_file, wav_scale=wav_scale)

        
        if rank == 0:
            logging.info("loading segments file %s" % segments_file)
        self.seg_set = SegmentSet.load(segments_file)
        if rank == 0:
            logging.info("dataset contains %d seqs" % len(self.seg_set))

        self.is_val = is_val
        if time_durs_file is not None:
            if rank == 0:
                logging.info("loading durations file %s" % time_durs_file)

            time_durs = SegmentSet.load(time_durs_file)
            self.seg_set["duration"] = time_durs.loc[
                self.seg_set["id"]
            ].class_id.values.astype(np.float, copy=False)
        else:
            assert "duration" in self.seg_set

        logging.info("loading class-info files")
        self._load_class_infos(class_names, class_files, is_val)


        if bpe_model is not None:
            logging.info("loading bpe models")
            self._load_bpe_model(bpe_model, is_val)

        if text_file is not None:
            logging.info("loading text files")
            self._load_text_infos(text_file, is_val)
        self.return_segment_info = (
            [] if return_segment_info is None else return_segment_info
        )
        self.return_orig = return_orig

        self.num_augs = num_augs
        self._create_augmenters(aug_cfgs)
        
        self.target_sample_freq = target_sample_freq
        self.resamplers = {}

    def _load_bpe_model(self, bpe_model, is_val):
        if self.rank == 0:
            logging.info("loading bpe file %s" % bpe_model)
        self.sp  = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        blank_id = self.sp.piece_to_id("<blk>")
        vocab_size = self.sp.get_piece_size()

    def _load_text_infos(self, text_file, is_val):
        if text_file is None:
            return
        if self.rank == 0:
            logging.info("loading text file %s" % text_file)
        
        text = read_text(text_file)
        self.seg_set["text"] = text.loc[self.seg_set["id"]].text

    def _load_class_infos(self, class_names, class_files, is_val):
        self.class_info = {}
        if class_names is None:
            assert class_files is None
            return

        assert len(class_names) == len(class_files)
        for name, file in zip(class_names, class_files):
            assert (
                name in self.seg_set
            ), f"class_name {name} not present in the segment set"
            if self.rank == 0:
                logging.info("loading class-info file %s" % file)
            table = ClassInfo.load(file)
            self.class_info[name] = table
            if not is_val:
                # check that all classes are present in the training segments
                class_ids = table["id"]
                segment_class_ids = self.seg_set[name].unique()
                for c_id in class_ids:
                    if c_id not in segment_class_ids:
                        logging.warning(
                            "%s class: %s not present in dataset", name, c_id
                        )

    def _create_augmenters(self, aug_cfgs):
        self.augmenters = []
        self.reverb_context = 0
        if aug_cfgs is None:
            return

        for aug_cfg in aug_cfgs:
            logging.info(f"loading augmentation={aug_cfg}")
            augmenter = SpeechAugment.create(
                aug_cfg, random_seed=112358 + 1000 * self.rank
            )
            self.augmenters.append(augmenter)
            self.reverb_context = max(augmenter.max_reverb_context, self.reverb_context)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def wav_scale(self):
        return self.r.wav_scale

    @property
    def num_seqs(self):
        return len(self.seg_set)

    def __len__(self):
        return self.num_seqs

    @property
    def seq_lengths(self):
        return self.seg_set["duration"]

    @property
    def total_length(self):
        return np.sum(self.seq_lengths)

    @property
    def min_seq_length(self):
        return np.min(self.seq_lengths)

    @property
    def max_seq_length(self):
        return np.max(self.seq_lengths)

    @property
    def num_classes(self):
        return {k: t.num_classes for k, t in self.class_info.items()}

    def _parse_segment_item(self, segment):
        if isinstance(segment, (tuple, list)):
            seg_id, start, duration = segment
            assert duration <= self.seg_set.loc[seg_id].duration, (
                f"{seg_id} with start={start} duration "
                f"({self.seg_set.loc[seg_id].duration}) < "
                f"chunk duration ({duration})"
            )
        else:
            seg_id, start, duration = segment, 0, 0

        if "start" in self.seg_set:
            start += self.seg_set.loc[seg_id].start

        return seg_id, start, duration

    def _read_audio(self, seg_id, start, duration):
        # how much extra audio we need to load to
        # calculate the reverb of the first part of the audio
        reverb_context = min(self.reverb_context, start)
        start -= reverb_context
        read_duration = duration + reverb_context

        # read audio
        recording_id = self.seg_set.recording_ids(seg_id)
        x, fs = self.r.read([recording_id], time_offset=start, time_durs=read_duration)
        return x[0].astype(floatstr_torch(), copy=False), fs[0]

    def _apply_augs(self, x, num_samples, reverb_context_samples):
        x_augs = []
        
        # for each type of augmentation
        for i, augmenter in enumerate(self.augmenters):
            # we do n_augs per augmentation type
            for j in range(self.num_augs):
                # augment x
                x_aug, aug_info = augmenter(x)
                # remove the extra left context used to compute the reverberation.
                x_aug = x_aug[reverb_context_samples : len(x)]
                x_augs.append(x_aug.astype(floatstr_torch(), copy=False))

        return x_augs

    def _get_segment_info(self, seg_id):
        r = []
        # converts the class_ids to integers
        for info_name in self.return_segment_info:
            seg_info = self.seg_set.loc[seg_id, info_name]
            if info_name in self.class_info:
                # if the type of information is a class-id
                # we use the class information table to
                # convert from id to integer
                class_info = self.class_info[info_name]
                idx = class_info.loc[seg_info, "class_idx"]
                seg_info = idx
            if info_name  == "text":
                seg_info = self.sp.encode(seg_info, out_type=int)

            r.append(seg_info)

        return r

    def _get_resampler(self, fs):
        if fs in self.resamplers:
            return self.resamplers[fs]

        resampler = tat.Resample(
            int(fs),
            int(self.target_sample_freq),
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492,
        )
        resampler_f = lambda x: resampler(torch.from_numpy(x)).numpy()
        self.resamplers[fs] = resampler_f
        return resampler_f

    def _resample(self, x, fs):
        try:
            if self.target_sample_freq is None or fs == self.target_sample_freq:
                return x, fs
            resampler = self._get_resampler(fs)
            return resampler(x), self.target_sample_freq
        except:
            return x, fs

    def __getitem__(self, segment):
        seg_id, start, duration = self._parse_segment_item(segment)
        x, fs = self._read_audio(seg_id, start, duration)
        x, fs = self._resample(x, fs)
        if self.augmenters:
            # augmentations
            if duration == 0:
                num_samples = len(x)
            else:
                num_samples = int(duration * fs)
            reverb_context_samples = len(x) - num_samples
            x_augs = self._apply_augs(x, num_samples, reverb_context_samples)
            
            r = x_augs

            # add original non augmented audio
            if self.return_orig:
                x_orig = x[reverb_context_samples:]
                r.append(x_orig)

        else:
            r = [x]

        # try:
        #     import soundfile as sf

        #     for i, z in enumerate(r):
        #         sf.write(f"file_{seg_id}.wav", z, fs, "PCM_16")
        # except:
        #     print("soundfile failed", flush=True)

        # adds the segment labels
        seg_info = self._get_segment_info(seg_id)
        r.extend(seg_info)

        return (*r,)

    @staticmethod
    def filter_args(**kwargs):

        ar_args = AR.filter_args(**kwargs)
        valid_args = (
            "audio_file",
            "segments_file",
            "aug_cfgs",
            "num_augs",
            "class_names",
            "class_files",
            "bpe_model",
            "text_file",
            "return_segment_info",
            "return_orig",
            "time_durs_file",
            "target_sample_freq",
        )
        args = dict((k, kwargs[k]) for k in valid_args if k in kwargs)
        args.update(ar_args)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip={}):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "audio_file" not in skip:
            parser.add_argument(
                "--audio-file", required=True, help=("audio manifest file"),
            )

        if "segments_file" not in skip:
            parser.add_argument(
                "--segments-file", required=True, help=("segments manifest file"),
            )

        parser.add_argument(
            "--class-names",
            default=None,
            nargs="+",
            help=(
                "list with the names of the types of classes in the datasets, e.g., speaker, language"
            ),
        )

        parser.add_argument(
            "--class-files", default=None, nargs="+", help=("list of class info files"),
        )

        parser.add_argument(
            "--time-durs-file",
            default=None,
            help=(
                "segment to duration in secs file, if durations are not in segments_file"
            ),
        )

        parser.add_argument(
            "--bpe-model",
            default=None,
            help=(
                "bpe model for the text label"
            ),
        )

        parser.add_argument(
            "--text-file",
            default=None,
            help=(
                "text file with words labels for each utterances"
            ),
        )

        parser.add_argument(
            "--aug-cfgs",
            default=None,
            nargs="+",
            help=("augmentation configuration file."),
        )

        parser.add_argument(
            "--num-augs",
            default=1,
            help=("number of augmentations per segment and augmentation type"),
        )
        parser.add_argument(
            "--return-segment-info",
            default=None,
            nargs="+",
            help=(
                "list of columns of the segment file which should be returned as supervisions"
            ),
        )
        parser.add_argument(
            "--return-orig",
            default=False,
            action=ActionYesNo,
            help=(
                "when using augmentation, whether or not to return also the original audio"
            ),
        )

        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help=(
                "target sampling frequencey, if not None all audios are converted to this sample freq"
            ),
        )

        AR.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='audio dataset options')

    add_argparse_args = add_class_args

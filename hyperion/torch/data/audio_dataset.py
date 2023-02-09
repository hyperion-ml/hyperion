"""
 Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time

import numpy as np
import pandas as pd
import torchaudio.transforms as tat
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from ...io import RandomAccessAudioReader as AR
from ...np.augment import SpeechAugment

#import k2
import sentencepiece as spm
#from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
import torch.distributed as dist

from ...utils.class_info import ClassInfo
from ...utils.segment_set import SegmentSet
from ...utils.text import read_text
from ..torch_defs import floatstr_torch


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
        wav_scale=2**15 - 1,
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
            logging.info("loading segments file %s", segments_file)

        self.seg_set = SegmentSet.load(segments_file)
        if rank == 0:
            logging.info("dataset contains %d seqs", len(self.seg_set))

        self.is_val = is_val
        if time_durs_file is not None:
            if rank == 0:
                logging.info("loading durations file %s", time_durs_file)

            time_durs = SegmentSet.load(time_durs_file)
            self.seg_set["duration"] = time_durs.loc[
                self.seg_set["id"]].class_id.values.astype(np.float,
                                                           copy=False)
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
        self.return_segment_info = ([] if return_segment_info is None else
                                    return_segment_info)
        self.return_orig = return_orig

        self.num_augs = num_augs
        self._create_augmenters(aug_cfgs)

        self.target_sample_freq = target_sample_freq
        self.resamplers = {}

    def _load_bpe_model(self, bpe_model, is_val):
        if self.rank == 0:
            logging.info("loading bpe file %s" % bpe_model)
        self.sp = spm.SentencePieceProcessor()
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
            assert (name in self.seg_set
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
                        logging.warning("%s class: %s not present in dataset",
                                        name, c_id)

    def _create_augmenters(self, aug_cfgs):
        self.augmenters = []
        self.reverb_context = 0
        if aug_cfgs is None:
            return

        for aug_cfg in aug_cfgs:
            logging.info(f"loading augmentation={aug_cfg}")
            augmenter = SpeechAugment.create(aug_cfg,
                                             random_seed=112358 +
                                             1000 * self.rank)
            self.augmenters.append(augmenter)
            self.reverb_context = max(augmenter.max_reverb_context,
                                      self.reverb_context)

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
                f"chunk duration ({duration})")
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
        x, fs = self.r.read([recording_id],
                            time_offset=start,
                            time_durs=read_duration)
        return x[0].astype(floatstr_torch(), copy=False), fs[0]

    def _apply_augs(self, x, num_samples, reverb_context_samples):
        x_augs = {}
        # for each type of augmentation
        for i, augmenter in enumerate(self.augmenters):
            # we do n_augs per augmentation type
            for j in range(self.num_augs):
                # augment x
                x_aug, aug_info = augmenter(x)
                # remove the extra left context used to compute the reverberation.
                x_aug = x_aug[reverb_context_samples:len(x)]
                x_aug = x_aug.astype(floatstr_torch(), copy=False)
                x_augs[f"x_aug_{i}_{j}"] = x_aug

        if not self.return_orig and len(x_augs) == 1:
            # if we just have one aug and we don't return the clean version,
            # we just call x to the aug version
            x_augs["x"] = x_augs.pop("x_aug_0_0")

        return x_augs

    def _get_segment_info(self, seg_id):
        seg_info = {}
        # converts the class_ids to integers
        for info_name in self.return_segment_info:
            seg_info_i = self.seg_set.loc[seg_id, info_name]
            if info_name in self.class_info:
                # if the type of information is a class-id
                # we use the class information table to
                # convert from id to integer
                class_info = self.class_info[info_name]
                seg_info_i = class_info.loc[seg_info_i, "class_idx"]

            if info_name == "text":
                seg_info_i = self.sp.encode(seg_info_i, out_type=int)

            seg_info[info_name] = seg_info_i

        return seg_info

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
        data = {"seg_id": seg_id, "sample_freq": fs}
        if self.augmenters:
            # augmentations
            if duration == 0:
                num_samples = len(x)
            else:
                num_samples = int(duration * fs)
            reverb_context_samples = len(x) - num_samples
            x_augs = self._apply_augs(x, num_samples, reverb_context_samples)
            data.update(x_augs)

            # add original non augmented audio
            if self.return_orig:
                x_orig = x[reverb_context_samples:]
                data["x"] = x_orig

        else:
            data["x"] = x

        # adds the segment labels
        seg_info = self._get_segment_info(seg_id)
        data.update(seg_info)
        return data

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
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "audio_file" not in skip:
            parser.add_argument(
                "--audio-file",
                required=True,
                help=("audio manifest file"),
            )

        if "segments_file" not in skip:
            parser.add_argument(
                "--segments-file",
                required=True,
                help=("segments manifest file"),
            )

        parser.add_argument(
            "--class-names",
            default=None,
            nargs="+",
            help=
            ("list with the names of the types of classes in the datasets, e.g., speaker, language"
             ),
        )

        parser.add_argument(
            "--class-files",
            default=None,
            nargs="+",
            help=("list of class info files"),
        )

        parser.add_argument(
            "--time-durs-file",
            default=None,
            help=
            ("segment to duration in secs file, if durations are not in segments_file"
             ),
        )

        parser.add_argument(
            "--bpe-model",
            default=None,
            help=("bpe model for the text label"),
        )

        parser.add_argument(
            "--text-file",
            default=None,
            help=("text file with words labels for each utterances"),
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
            help=
            ("list of columns of the segment file which should be returned as supervisions"
             ),
        )
        parser.add_argument(
            "--return-orig",
            default=False,
            action=ActionYesNo,
            help=
            ("when using augmentation, whether or not to return also the original audio"
             ),
        )

        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help=
            ("target sampling frequencey, if not None all audios are converted to this sample freq"
             ),
        )

        AR.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix,
                                      action=ActionParser(parser=parser))
            # help='audio dataset options')

    add_argparse_args = add_class_args

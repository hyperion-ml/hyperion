"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# import k2
import sentencepiece as spm
import torch
import torch.distributed as dist
import torchaudio.transforms as tat
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...io import RandomAccessAudioReader as AR
from ...np.augment import SpeechAugment
from ...utils.class_info import ClassInfo
from ...utils.misc import filter_func_args
from ...utils.segment_set import SegmentSet
from ...utils.text import read_text
from ..torch_defs import floatstr_torch
from .audio_dataset import AudioDataset


class DINOAudioDataset(AudioDataset):
    """AudioDataset class to train DINO for speech

    Args:
      recordings_file: recordings manifest file (kaldi .scp or pandas .csv)
      segments_file: segments manifest file (kaldi .scp or pandas .csv)
      class_names: list with the names of the types of classes in the datasets, e.g., speaker, language
      class_files: list of class info files
      time_durs_file: (deprecated) segment to duration in secs file, if durations are not in segments_file
      bpe_model: bpe model for the text label
      text_file: text file with words labels for each utterances
      teacher_aug_cfg: configuration for teacher augmentations
      student_aug_cfg: configuration for student augmentations.
      aug_cfgs: list of augmentation configuration files
      num_augs: number of augmentations per segment and augmentation type
      num_aug_mix: "number of AugMix augmentations per segment
      aug_mix_alpha: AugMix Diritchlet distribution parameter
      return_segment_info: list of columns of the segment file which should be returned as supervisions
      return_orig: when using augmentation, whether or not to return also the original audio
      target_sample_freq: target sampling frequencey, if not None all audios are converted to this sample freq
      wav_scale: make waves to be in [-wav_scale, wav_scale]
      is_val: is validation dataset.
      seed: random seed
      teacher_chunk_length: chunk length for the teacher model
      num_teacher_chunks: num teacher chunks in eachd batch
      student_chunk_length: chunk length for the student model
      num_student_chunks: num student chunks in eachd batch
      same_teacher_student_chunks: is True if teacher and student chunks are overlapped, False if disjoint
    """

    def __init__(
        self,
        recordings_file: str,
        segments_file: str,
        class_names: Optional[List[str]] = None,
        class_files: Optional[List[str]] = None,
        bpe_model: Optional[str] = None,
        text_file: Optional[str] = None,
        time_durs_file: Optional[str] = None,
        teacher_aug_cfg: Optional[str] = None,
        student_aug_cfg: Optional[str] = None,
        num_augs: int = 1,
        num_aug_mix: int = 0,
        aug_mix_alpha: float = 0,
        return_segment_info: Optional[List[str]] = None,
        return_orig: bool = False,
        target_sample_freq: Optional[float] = None,
        wav_scale: float = 1,
        is_val: bool = False,
        seed: int = 112358,
        teacher_chunk_length: float = 4,
        num_teacher_chunks: int = 2,
        student_chunk_length: float = 2,
        num_student_chunks: int = 4,
        same_teacher_student_chunks: bool = False,
    ):
        aug_cfgs = []
        student_aug_idx = -1
        teacher_aug_idx = -1
        if student_aug_cfg is not None:
            aug_cfgs.append(student_aug_cfg)
            student_aug_idx = 0
        if teacher_aug_cfg is not None:
            assert student_aug_idx is not None
            if teacher_aug_cfg != student_aug_cfg:
                aug_cfgs.append(teacher_aug_cfg)
                teacher_aug_idx = 1
            else:
                teacher_aug_idx = 0

        super_args = filter_func_args(super().__init__, locals())
        super().__init__(**super_args)
        self.teacher_chunk_length = teacher_chunk_length
        self.num_teacher_chunks = num_teacher_chunks
        self.student_chunk_length = student_chunk_length
        self.num_student_chunks = num_student_chunks
        self.same_teacher_student_chunks = same_teacher_student_chunks
        if student_aug_idx != -1:
            self.student_augmenter = self.augmenters[student_aug_idx]
        if teacher_aug_idx != -1:
            self.teacher_augmenter = self.augmenters[teacher_aug_idx]

    def _apply_chunk_augs(self, x, duration, fs, augmenter, tag):
        if not augmenter:
            return {f"x_{tag}": x}

        if duration == 0:
            num_samples = len(x)
        else:
            num_samples = int(duration * fs)

        reverb_context_samples = len(x) - num_samples
        x_orig = x[reverb_context_samples:]
        x_augs = {}
        for j in range(self.num_augs):
            # augment x
            x_aug, aug_info = augmenter(x)
            # remove the extra left context used to compute the reverberation.
            x_aug = x_aug[reverb_context_samples : len(x)]
            x_aug = x_aug.astype(floatstr_torch(), copy=False)
            x_augs[f"x_{tag}_aug_{j}"] = x_aug

        if self.num_aug_mix > 0:
            x_augs = self._apply_aug_mix(x_orig, x_augs, 0)

        if self.return_orig:
            x_augs[f"x_{tag}"] = x_orig
        elif len(x_augs) == 1:
            # if we just have one aug and we don't return the clean version,
            # we just call x to the aug version
            x_augs[f"x_{tag}"] = x_augs.pop(f"x_{tag}_aug_0")

        return x_augs

    def _apply_augs(self, xs, duration, fs, augmenter, tag):
        x_augs = {}
        for i, x in enumerate(xs):
            x_augs_i = self._apply_chunk_augs(x, duration, fs, augmenter, f"{tag}_{i}")
            x_augs.update(x_augs_i)

        return x_augs

    def _split_audio_into_chunks(self, x, x_samples, chunk_samples, num_chunks):
        reverb_context = len(x) - x_samples
        chunk_shift = (x_samples - chunk_samples) // num_chunks
        xs = []
        for i in range(num_chunks):
            x_start = i * chunk_shift
            x_end = x_start + chunk_samples + reverb_context
            xs.append(x[x_start:x_end])

        return xs

    def _split_audio_into_teacher_student_disjoint(self, x, duration, fs):
        total_samples = int(duration * fs)
        teacher_chunk_samples = int(fs * self.teacher_chunk_length)
        student_chunk_samples = int(fs * self.student_chunk_length)
        sum_chunk = teacher_chunk_samples + student_chunk_samples
        assert total_samples >= sum_chunk, f"signal samples = {len(x)} < {sum_chunk}"

        teacher_crops_x_chunk = self.num_teacher_chunks * teacher_chunk_samples
        student_crops_x_chunk = self.num_student_chunks * student_chunk_samples
        sum_crops_x_chunk = teacher_crops_x_chunk + student_crops_x_chunk
        teacher_samples = max(
            teacher_crops_x_chunk * total_samples // sum_crops_x_chunk,
            teacher_chunk_samples,
        )
        student_samples = total_samples - teacher_samples
        # here we decide if we split the audio in [teacher, student] or [student, teacher]
        teacher_first = self.rng.random() < 0.5

        if teacher_first:
            x1_samples = teacher_samples
            # x2_samples = student_samples
        else:
            x1_samples = student_samples
            # x2_samples = teacher_samples

        max_reverb_context = int(self.reverb_context * fs)
        x1_reverb_context = len(x) - total_samples
        x1_end_sample = x1_reverb_context + x1_samples
        x1 = x[:x1_end_sample]
        if x1_end_sample >= max_reverb_context:
            x2_reverb_context = max_reverb_context
        else:
            x2_reverb_context = x1_end_sample

        # print(
        #     "xxx",
        #     len(x),
        #     total_samples,
        #     teacher_first,
        #     teacher_samples,
        #     student_samples,
        #     x1_reverb_context,
        #     x1_end_sample,
        #     x2_reverb_context,
        #     flush=True,
        # )
        x2 = x[x1_end_sample - x2_reverb_context :]
        if teacher_first:
            x_teacher = x1
            x_student = x2
        else:
            x_teacher = x2
            x_student = x1

        return x_teacher, teacher_samples, x_student, student_samples

    def _split_audio_into_teacher_student_same(self, x, duration, fs):
        total_samples = int(duration * fs)
        return x, total_samples, x, total_samples

    def _split_audio_into_teacher_student_chunks(self, x, duration, fs):
        if self.same_teacher_student_chunks:
            (
                x_teacher,
                teacher_samples,
                x_student,
                student_samples,
            ) = self._split_audio_into_teacher_student_same(x, duration, fs)
        else:
            (
                x_teacher,
                teacher_samples,
                x_student,
                student_samples,
            ) = self._split_audio_into_teacher_student_disjoint(x, duration, fs)
        assert (
            len(x_teacher) >= 64000 and len(x_teacher) <= 136000
        ), f"{len(x_teacher)}, {len(x_student)} {len(x)} {duration*fs}, {teacher_samples}, {student_samples}"
        assert (
            len(x_student) >= 32000 and len(x_student) <= 136000
        ), f"{len(x_teacher)}, {len(x_student)}, {len(x)} {duration*fs}, {teacher_samples}, {student_samples}"
        xs_teacher = self._split_audio_into_chunks(
            x_teacher,
            teacher_samples,
            int(fs * self.teacher_chunk_length),
            self.num_teacher_chunks,
        )
        xs_student = self._split_audio_into_chunks(
            x_student,
            student_samples,
            int(fs * self.student_chunk_length),
            self.num_student_chunks,
        )
        for xx in xs_teacher:
            assert (
                len(xx) >= 64000 and len(xx) <= 72000
            ), f"{[len(t) for t in xs_teacher]} {len(x_teacher)} {len(x)}"
        for xx in xs_student:
            assert (
                len(xx) >= 32000 and len(xx) <= 40000
            ), f"{[len(t) for t in xs_student]} {len(x_student)} {len(x)}"

        return xs_teacher, xs_student

    def __getitem__(self, segment):
        seg_id, start, duration = self._parse_segment_item(segment)
        x, fs = self._read_audio(seg_id, start, duration)
        x, fs = self._resample(x, fs)
        assert len(x) >= int(
            duration * fs
        ), f"getitem {self.seg_set.loc[seg_id].duration}, {start}, {duration}, {len(x)}"
        data = {"seg_id": seg_id, "sample_freq": fs}
        xs_teacher, xs_student = self._split_audio_into_teacher_student_chunks(
            x, duration, fs
        )
        x_augs_teacher = self._apply_augs(
            xs_teacher, self.teacher_chunk_length, fs, self.teacher_augmenter, "teacher"
        )
        x_augs_student = self._apply_augs(
            xs_student, self.student_chunk_length, fs, self.student_augmenter, "student"
        )
        data.update(x_augs_teacher)
        data.update(x_augs_student)
        # print(data, flush=True)
        # for ll in [
        #     "x_teacher_0",
        #     "x_teacher_1",
        #     "x_student_0",
        #     "x_student_1",
        #     "x_student_2",
        #     "x_student_3",
        # ]:
        #     print("zzz ", ll, data[ll].shape, flush=True)
        seg_info = self._get_segment_info(seg_id)
        data.update(seg_info)
        return data

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(DINOAudioDataset.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip.add("aug_cfgs")
        AudioDataset.add_class_args(parser, skip=skip)
        parser.add_argument(
            "--teacher-aug-cfg", default=None, help="config for teacher augmentations"
        )
        parser.add_argument(
            "--student-aug-cfg", default=None, help="config for student augmentations"
        )
        parser.add_argument(
            "--teacher-chunk-length",
            default=4.0,
            type=float,
            help="chunk length for the teacher model",
        )
        parser.add_argument(
            "--student-chunk-length",
            default=4.0,
            type=float,
            help="chunk length for the student model",
        )
        parser.add_argument(
            "--num-teacher-chunks",
            default=2,
            type=int,
            help="num teacher chunks in eachd batch",
        )
        parser.add_argument(
            "--num-student-chunks",
            default=4,
            type=int,
            help="num student chunks in eachd batch",
        )
        parser.add_argument(
            "--same-teacher-student-chunks",
            default=False,
            action=ActionYesNo,
            help="teacher and student chunks are overlapped",
        )

        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

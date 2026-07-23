"""
Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser

from ...np.augment import SpeechAugment
from ...utils.misc import filter_func_args
from ..torch_defs import floatstr_torch
from .legacy_audio_dataset import LegacyAudioDataset


class DINOAudioDataset(LegacyAudioDataset):
    """Audio dataset that returns teacher and student chunks for DINO training.

    Args:
      recordings_file: Recordings manifest file.
      segments_file: Segment metadata file.
      class_names: Segment columns to convert from class ids to class indexes.
      class_files: Class information files aligned with class_names.
      tokenizer_mappings: Mappings from segment fields to tokenizer aliases,
            formatted as input->tokenizer.
      tokenizer_files: Tokenizer configuration files aligned with
            tokenizer_mappings.
      text_file: Deprecated text file with word labels for utterances.
      time_durs_file: Deprecated segment-duration file.
      teacher_aug_cfg: Configuration for teacher augmentations.
      student_aug_cfg: Configuration for student augmentations.
      num_augs: Number of augmentations per chunk and augmentation type.
      num_aug_mix: Number of AugMix augmentations per chunk.
      aug_mix_alpha: AugMix Dirichlet and beta distribution parameter.
      return_segment_info: Segment columns or tokenizer aliases returned as
            supervisions.
      return_orig: Whether to return the original chunk when augmentation is
            enabled.
      target_sample_freq: Target sample frequency. If not None, audio is
            resampled to this frequency.
      wav_scale: Multiplicative factor for waveform samples.
      is_val: Whether this is a validation dataset.
      enable_tel_codecs_if: SegmentSet expression controlling telephone codec
            augmentation.
      enable_media_codecs_if: SegmentSet expression controlling media codec
            augmentation.
      enable_transcodec_if: SegmentSet expression controlling transcodec
            augmentation.
      seed: Random seed.
      teacher_chunk_length: Chunk length for the teacher model in seconds.
      num_teacher_chunks: Number of teacher chunks in each item.
      student_chunk_length: Chunk length for the student model in seconds.
      num_student_chunks: Number of student chunks in each item.
      same_teacher_student_chunks: Whether teacher and student chunks are
            overlapped instead of disjoint.

    Attributes:
      teacher_chunk_length: Chunk length for the teacher model in seconds.
      num_teacher_chunks: Number of teacher chunks in each item.
      student_chunk_length: Chunk length for the student model in seconds.
      num_student_chunks: Number of student chunks in each item.
      same_teacher_student_chunks: Whether teacher and student chunks overlap.
      teacher_augmenter: Optional augmenter used for teacher chunks.
      student_augmenter: Optional augmenter used for student chunks.
    """

    def __init__(
        self,
        recordings_file: str,
        segments_file: str,
        class_names: Optional[List[str]] = None,
        class_files: Optional[List[str]] = None,
        tokenizer_mappings: Optional[List[str]] = None,
        tokenizer_files: Optional[List[str]] = None,
        text_file: Optional[str] = None,
        time_durs_file: Optional[str] = None,
        teacher_aug_cfg: Optional[str] = None,
        student_aug_cfg: Optional[str] = None,
        num_augs: int = 1,
        num_aug_mix: int = 0,
        aug_mix_alpha: float = 0.5,
        return_segment_info: Optional[List[str]] = None,
        return_orig: bool = False,
        target_sample_freq: Optional[float] = None,
        wav_scale: float = 1,
        is_val: bool = False,
        enable_tel_codecs_if: Optional[str] = None,
        enable_media_codecs_if: Optional[str] = None,
        enable_transcodec_if: Optional[str] = None,
        seed: int = 112358,
        teacher_chunk_length: float = 4,
        num_teacher_chunks: int = 2,
        student_chunk_length: float = 2,
        num_student_chunks: int = 4,
        same_teacher_student_chunks: bool = False,
    ) -> None:
        """Initializes a DINOAudioDataset.

        Args:
          recordings_file: Recordings manifest file.
          segments_file: Segment metadata file.
          class_names: Segment columns to convert from class ids to class
            indexes.
          class_files: Class information files aligned with class_names.
          tokenizer_mappings: Mappings formatted as input->tokenizer.
          tokenizer_files: Tokenizer configuration files aligned with
            tokenizer_mappings.
          text_file: Deprecated text file with word labels for utterances.
          time_durs_file: Deprecated segment-duration file.
          teacher_aug_cfg: Configuration for teacher augmentations.
          student_aug_cfg: Configuration for student augmentations.
          num_augs: Number of augmentations per chunk and augmentation type.
          num_aug_mix: Number of AugMix augmentations per chunk.
          aug_mix_alpha: AugMix Dirichlet and beta distribution parameter.
          return_segment_info: Segment columns or tokenizer aliases returned as
            supervisions.
          return_orig: Whether to return the original chunk when augmentation
            is enabled.
          target_sample_freq: Optional target sample frequency.
          wav_scale: Multiplicative factor for waveform samples.
          is_val: Whether this is a validation dataset.
          enable_tel_codecs_if: SegmentSet expression controlling telephone
            codec augmentation.
          enable_media_codecs_if: SegmentSet expression controlling media codec
            augmentation.
          enable_transcodec_if: SegmentSet expression controlling transcodec
            augmentation.
          seed: Random seed.
          teacher_chunk_length: Chunk length for the teacher model in seconds.
          num_teacher_chunks: Number of teacher chunks in each item.
          student_chunk_length: Chunk length for the student model in seconds.
          num_student_chunks: Number of student chunks in each item.
          same_teacher_student_chunks: Whether teacher and student chunks are
            overlapped instead of disjoint.
        """
        aug_cfgs = []
        student_aug_idx = -1
        teacher_aug_idx = -1
        if student_aug_cfg is not None:
            aug_cfgs.append(student_aug_cfg)
            student_aug_idx = 0
        if teacher_aug_cfg is not None:
            if student_aug_idx == -1:
                aug_cfgs.append(teacher_aug_cfg)
                teacher_aug_idx = 0
            elif teacher_aug_cfg != student_aug_cfg:
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
        self.student_augmenter = None
        self.teacher_augmenter = None
        if student_aug_idx != -1:
            self.student_augmenter = self.augmenters[student_aug_idx]
        if teacher_aug_idx != -1:
            self.teacher_augmenter = self.augmenters[teacher_aug_idx]

    def _apply_chunk_augs(
        self,
        x: np.ndarray,
        duration: float,
        fs: int,
        augmenter: Optional[SpeechAugment],
        tag: str,
        enable_codecs: Dict[str, bool],
    ) -> Dict[str, np.ndarray]:
        """Applies augmentation to one teacher or student chunk.

        Args:
          x: Input waveform chunk.
          duration: Chunk duration in seconds.
          fs: Sample frequency.
          augmenter: Optional speech augmenter.
          tag: Output key tag.
          enable_codecs: Codec augmentation switches.

        Returns:
          Dictionary containing augmented or original waveform chunks.
        """
        if duration == 0:
            num_samples = len(x)
        else:
            num_samples = int(duration * fs)

        reverb_context_samples = len(x) - num_samples
        x_orig = x[reverb_context_samples:]
        if not augmenter:
            return {f"x_{tag}": x_orig}

        x_augs = {}
        for j in range(self.num_augs):
            # augment x
            x_aug, aug_info = augmenter(x, fs, **enable_codecs)
            # remove the extra left context used to compute the reverberation.
            x_aug = x_aug[reverb_context_samples : len(x)]
            x_aug = x_aug.astype(floatstr_torch(), copy=False)
            x_augs[f"x_{tag}_aug_{j}"] = x_aug

        if self.num_aug_mix > 0:
            x_aug_mix = self._apply_aug_mix(x_orig, x_augs, 0)
            x_augs = {
                key.replace("x_aug_0", f"x_{tag}_aug_mix"): value
                for key, value in x_aug_mix.items()
            }

        if self.return_orig:
            x_augs[f"x_{tag}"] = x_orig
        elif len(x_augs) == 1:
            # if we just have one aug and we don't return the clean version,
            # we just call x to the aug version
            x_augs[f"x_{tag}"] = x_augs.pop(f"x_{tag}_aug_0")

        return x_augs

    def _apply_augs(
        self,
        xs: List[np.ndarray],
        duration: float,
        fs: int,
        augmenter: Optional[SpeechAugment],
        tag: str,
        enable_codecs: Dict[str, bool],
    ) -> Dict[str, np.ndarray]:
        """Applies augmentation to a list of teacher or student chunks.

        Args:
          xs: Waveform chunks.
          duration: Chunk duration in seconds.
          fs: Sample frequency.
          augmenter: Optional speech augmenter.
          tag: Output key tag.
          enable_codecs: Codec augmentation switches.

        Returns:
          Dictionary containing waveform chunks keyed by chunk role and index.
        """
        x_augs = {}
        for i, x in enumerate(xs):
            x_augs_i = self._apply_chunk_augs(
                x, duration, fs, augmenter, f"{tag}_{i}", enable_codecs
            )
            x_augs.update(x_augs_i)

        return x_augs

    def _split_audio_into_chunks(
        self, x: np.ndarray, x_samples: int, chunk_samples: int, num_chunks: int
    ) -> List[np.ndarray]:
        """Splits an audio span into fixed-length chunks.

        Args:
          x: Input waveform with optional left reverb context.
          x_samples: Number of non-context samples in x.
          chunk_samples: Number of non-context samples per chunk.
          num_chunks: Number of chunks to return.

        Returns:
          List of waveform chunks.
        """
        reverb_context = len(x) - x_samples
        chunk_shift = (x_samples - chunk_samples) // num_chunks
        xs = []
        for i in range(num_chunks):
            x_start = i * chunk_shift
            x_end = x_start + chunk_samples + reverb_context
            xs.append(x[x_start:x_end])

        return xs

    def _split_audio_into_teacher_student_disjoint(
        self, x: np.ndarray, duration: float, fs: int
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        """Splits audio into disjoint teacher and student spans.

        Args:
          x: Input waveform with optional left reverb context.
          duration: Requested segment duration in seconds.
          fs: Sample frequency.

        Returns:
          Teacher waveform, teacher sample count, student waveform, and student
          sample count.
        """
        total_samples = len(x) if duration == 0 else int(duration * fs)
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

    def _split_audio_into_teacher_student_same(
        self, x: np.ndarray, duration: float, fs: int
    ) -> Tuple[np.ndarray, int, np.ndarray, int]:
        """Uses the same audio span for teacher and student chunks.

        Args:
          x: Input waveform with optional left reverb context.
          duration: Requested segment duration in seconds.
          fs: Sample frequency.

        Returns:
          Teacher waveform, teacher sample count, student waveform, and student
          sample count.
        """
        total_samples = len(x) if duration == 0 else int(duration * fs)
        return x, total_samples, x, total_samples

    def _split_audio_into_teacher_student_chunks(
        self, x: np.ndarray, duration: float, fs: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Splits audio into teacher and student chunk lists.

        Args:
          x: Input waveform with optional left reverb context.
          duration: Requested segment duration in seconds.
          fs: Sample frequency.

        Returns:
          Teacher chunks and student chunks.
        """
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
        # assert (
        #     len(x_teacher) >= 64000 and len(x_teacher) <= 136000
        # ), (
        #     f"{len(x_teacher)}, {len(x_student)} {len(x)} {duration*fs}, "
        #     f"{teacher_samples}, {student_samples}"
        # )
        # assert (
        #     len(x_student) >= 32000 and len(x_student) <= 136000
        # ), (
        #     f"{len(x_teacher)}, {len(x_student)}, {len(x)} {duration*fs}, "
        #     f"{teacher_samples}, {student_samples}"
        # )
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
        # for xx in xs_teacher:
        #     assert (
        #         len(xx) >= 64000 and len(xx) <= 72000
        #     ), f"{[len(t) for t in xs_teacher]} {len(x_teacher)} {len(x)}"
        # for xx in xs_student:
        #     assert (
        #         len(xx) >= 32000 and len(xx) <= 40000
        #     ), f"{[len(t) for t in xs_student]} {len(x_student)} {len(x)}"

        return xs_teacher, xs_student

    def __getitem__(
        self, segment: Union[str, Tuple[str, float, float], List[Any]]
    ) -> Dict[str, Any]:
        """Reads one DINO training item.

        Args:
          segment: Segment id, or tuple/list with segment id, start, and
            duration.

        Returns:
          Dictionary containing teacher/student chunks and segment metadata.
        """
        self._maybe_reseed_worker()
        seg_id, start, duration = self._parse_segment_item(segment)
        x, fs = self._read_audio(seg_id, start, duration)
        x, fs = self._resample(x, fs)
        assert len(x) >= int(
            duration * fs
        ), f"getitem {self.seg_set.loc[seg_id].duration}, {start}, {duration}, {len(x)}"
        data = {"seg_id": seg_id, "sample_freq": fs}
        enable_codecs = self._get_enable_codecs(seg_id)
        xs_teacher, xs_student = self._split_audio_into_teacher_student_chunks(
            x, duration, fs
        )
        x_augs_teacher = self._apply_augs(
            xs_teacher,
            self.teacher_chunk_length,
            fs,
            self.teacher_augmenter,
            "teacher",
            enable_codecs,
        )
        x_augs_student = self._apply_augs(
            xs_student,
            self.student_chunk_length,
            fs,
            self.student_augmenter,
            "student",
            enable_codecs,
        )
        data.update(x_augs_teacher)
        data.update(x_augs_student)
        seg_info = self._get_segment_info(seg_id)
        data.update(seg_info)
        return data

    @staticmethod
    def filter_args(**kwargs: Any) -> Dict[str, Any]:
        """Filters keyword arguments accepted by the constructor.

        Args:
          **kwargs: Candidate keyword arguments.

        Returns:
          Constructor keyword arguments.
        """
        args = filter_func_args(DINOAudioDataset.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(
        parser: ArgumentParser, prefix: Optional[str] = None, skip: set = set()
    ) -> None:
        """Adds dataset constructor arguments to an argument parser.

        Args:
          parser: Argument parser.
          prefix: Optional nested parser prefix.
          skip: Argument names to skip.
        """
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        skip = set(skip)
        skip.add("aug_cfgs")
        LegacyAudioDataset.add_class_args(parser, skip=skip)
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
            default=2.0,
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

"""
Copyright 2020 Johns Hopkins University  (Author: Jesus Villalba)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# import k2
try:
    import k2
except:
    from ..utils import dummy_k2 as k2

import sentencepiece as spm
import torch
import torch.distributed as dist
import torchaudio.transforms as tat
from jsonargparse import ActionParser, ActionYesNo, ArgumentParser
from torch.utils.data import Dataset

from ...io import RandomAccessAudioReader as AR
from ...np.augment import SpeechAugment
from ...np.preprocessing import ResamplerToTargetFreq
from ...utils import ClassInfo, HyperDataset, PathLike, SegmentSet
from ...utils.misc import filter_func_args
from ...utils.text import read_text
from ..tokenizers import HypTokenizer
from ..torch_defs import floatstr_torch
from ..utils import collate_seqs_1d, collate_seqs_nd, list_of_dicts_to_list


class AudioDataset(Dataset):
    """AudioDataset class

    Args:
      recordings_file: recordings manifest file (kaldi .scp or pandas .csv)
      segments_file: segments manifest file (kaldi .scp or pandas .csv)
      class_names: list with the names of the types of classes in the datasets, e.g., speaker, language
      class_files: list of class info files
      tokenizer_mappings: list mapping the segment_set fields to the tokenizer name
            that should be used with them, e.g., text->text-1,
            this argument has to be sync with tokenizer_files.
      tokenizer_files: list of tokenizer cofinguration files
            this argument has to be sync with tokenizer_mappings.
      aug_cfgs: list of augmentation configuration files
      num_augs: number of augmentations per segment and augmentation type
      num_aug_mix: "number of AugMix augmentations per segment
      aug_mix_alpha: AugMix Diritchlet distribution parameter
      return_segment_info: list of columns of the segment file which should be returned as supervisions
      return_orig: when using augmentation, whether or not to return also the original audio
      target_sample_freq: target sampling frequencey, if not None all audios are converted to this sample freq
      wav_scale: make waves to be in [-wav_scale, wav_scale]
      is_val: is validation dataset.
      enable_tel_codecs_if: condition to enable telephone codec augmentation,
            for example use only if the segment is not conv. tel. speech: source_type != 'cts'
      enable_media_codecs_if: condition to enable media codec augmentation,
            for example use only if the segment is audio from video: source_type == 'afv'
      enable_transcodecs_if: condition to enable transcodec augmentation,
            for example use transcodec only if the segment is spoof: spoof_det == 'spoof'
      seed: random seed",
      time_durs_file: (deprecated) segment to duration in secs file, if durations are not in segments_file
      text_file: (deprecated) text file with words labels for each utterances.
      bpe_model: (deprecated) bpe model for the text label.
    """

    def __init__(
        self,
        dataset_path: PathLike,
        class_names: Optional[List[str]] = None,
        tokenizer_mappings: Optional[List[str]] = None,
        tokenizer_files: Optional[List[str]] = None,
        extra_attrs: Optional[List[str]] = None,
        aug_cfgs: Optional[List[str]] = None,
        num_augs: int = 1,
        num_aug_mix: int = 0,
        aug_mix_alpha: float = 0,
        target_sample_freq: Optional[float] = None,
        wav_scale: float = 1.0,
        is_val: bool = False,
        enable_tel_codecs_if: Optional[str] = None,
        enable_media_codecs_if: Optional[str] = None,
        enable_transcodec_if: Optional[str] = None,
        seed: int = 112358,
        bpe_model: Optional[str] = None,
    ):
        super().__init__()
        try:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        except:
            rank = 0
            world_size = 1

        if rank == 0:
            logging.info("loading dataset %s", dataset_path)

        self.dataset = HyperDataset.load(dataset_path)

        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        recordings = self.dataset.recordings()
        segments = self.dataset.segments()
        self.r = AR(
            recordings=recordings,
            segments=segments if segments.has_time_marks else None,
            wav_scale=wav_scale,
        )

        self.is_val = is_val
        self.class_names = class_names if class_names is not None else []
        self.exta_attrs = extra_attrs if extra_attrs is not None else []

        # logging.info("loading class-info files")
        # self._load_class_infos(class_names, class_files, is_val)

        self._load_tokenizers(tokenizer_mappings, tokenizer_files)

        if bpe_model is not None:
            logging.info("loading bpe models")
            self._load_bpe_model(bpe_model)

        self.num_augs = num_augs
        self.num_aug_mix = num_aug_mix
        self.aug_mix_alpha = aug_mix_alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000 * rank)
        self._create_augmenters(aug_cfgs)
        self._worker_reseeded = False

        self.target_sample_freq = target_sample_freq
        # self.resamplers = {}
        self.resampler = ResamplerToTargetFreq(target_sample_freq)

        # prepare enable codecs conditions
        self.enable_tel_codecs_if = enable_tel_codecs_if
        self.enable_media_codecs_if = enable_media_codecs_if
        self.enable_transcodec_if = enable_transcodec_if
        if enable_tel_codecs_if:
            segments.eval(f"enable_tel_codecs = {enable_tel_codecs_if}", inplace=True)
        if enable_media_codecs_if:
            segments.eval(
                f"enable_media_codecs = {enable_media_codecs_if}", inplace=True
            )
        if enable_transcodec_if:
            segments.eval(f"enable_transcodec = {enable_transcodec_if}", inplace=True)

    def _load_bpe_model(self, bpe_model):
        if self.rank == 0:
            logging.info("loading bpe file %s", bpe_model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        blank_id = self.sp.piece_to_id("<blk>")
        vocab_size = self.sp.get_piece_size()

    # def _load_class_infos(self, class_names, is_val):
    #     self.class_info = OrderedDict()
    #     if class_names is None:
    #         assert class_files is None
    #         return

    #     assert len(class_names) == len(class_files)
    #     for name, file in zip(class_names, class_files):
    #         assert (
    #             name in self.seg_set
    #         ), f"class_name {name} not present in the segment set"
    #         self.seg_set.convert_col_to_str(
    #             name
    #         )  # make sure that class ids are strings
    #         if self.rank == 0:
    #             logging.info("loading class-info file %s", file)
    #         table = ClassInfo.load(file)
    #         self.class_info[name] = table
    #         if not is_val:
    #             # check that all classes are present in the training segments
    #             class_ids = table["id"]
    #             segment_class_ids = self.seg_set[name].unique()
    #             for c_id in class_ids:
    #                 if c_id not in segment_class_ids:
    #                     logging.warning(
    #                         "%s class: %s not present in dataset", name, c_id
    #                     )

    def _load_tokenizers(self, tokenizer_files, tokenizer_mappings):
        self.tokenizers = OrderedDict()
        self.output_attrs_to_input = OrderedDict()
        self.output_attrs_to_tokenizers = OrderedDict()

        if tokenizer_mappings is None:
            assert tokenizer_files is None
            return

        logging.info("loading tokenizers")

        for tokenizer_file in tokenizer_files:
            assert isinstance(
                tokenizer_file, str
            ), f"tokenizer file {tokenizer_file} should be a string"
            tokenizer_name, tokenizer_file = tokenizer_file.split(":", maxsplit=1)
            if self.rank == 0:
                logging.info("loading tokenizer file %s", tokenizer_file)
            tokenizer = HypTokenizer.auto_load(tokenizer_file)
            self.tokenizers[tokenizer_name] = tokenizer

        for map in tokenizer_mappings:
            in_attr_name, tokenizer_name, out_attr_name = map.split("->", maxsplit=2)
            self.output_attrs_to_input[out_attr_name] = in_attr_name
            self.output_attrs_to_tokenizers[out_attr_name] = tokenizer_name

    def _create_augmenters(self, aug_cfgs):
        self.augmenters = []
        self.reverb_context = 0
        if aug_cfgs is None:
            return

        for aug_cfg in aug_cfgs:
            logging.info(f"loading augmentation={aug_cfg}")
            augmenter = SpeechAugment.create(
                aug_cfg, random_seed=self.seed + 1000 * self.rank
            )
            self.augmenters.append(augmenter)
            self.reverb_context = max(augmenter.max_reverb_context, self.reverb_context)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _maybe_reseed_worker(self):
        if self._worker_reseeded:
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self._worker_reseeded = True
            return

        base_seed = int(worker_info.seed) + 1000 * self.rank
        self.rng = np.random.default_rng(seed=base_seed)
        child_seeds = np.random.SeedSequence(base_seed).spawn(len(self.augmenters))
        for i, augmenter in enumerate(self.augmenters):
            if hasattr(augmenter, "reseed"):
                augmenter.reseed(child_seeds[i])

        self._worker_reseeded = True

    @property
    def segments(self):
        """Returns the segments dataframe."""
        return self.dataset.segments()

    @property
    def wav_scale(self):
        return self.r.wav_scale

    @property
    def num_seqs(self):
        return len(self.dataset)

    def __len__(self):
        return self.num_seqs

    @property
    def seq_lengths(self):
        return self.dataset.segments()["duration"]

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
        return {k: self.dataset.classes_value(k).num_classes for k in self.class_names}

    @property
    def class_info(self):
        return {k: self.dataset.classes_value(k) for k in self.class_names}

    def _parse_segment_item(self, segment):
        if isinstance(segment, (tuple, list)):
            seg_id, start, duration = segment
            assert duration <= self.dataset.segments().loc[seg_id].duration, (
                f"{seg_id} with start={start} duration "
                f"({self.seg_set.loc[seg_id].duration}) < "
                f"chunk duration ({duration})"
            )
        else:
            seg_id, start, duration = segment, 0, 0

        return seg_id, start, duration

    def _read_audio(self, seg_id, start, duration):
        # how much extra audio we need to load to
        # calculate the reverb of the first part of the audio
        reverb_context = min(self.reverb_context, start)
        start -= reverb_context
        read_duration = duration + reverb_context

        # read audio
        x, fs = self.r.read([seg_id], time_offset=start, time_durs=read_duration)
        x = x[0].astype(floatstr_torch(), copy=False)
        if x.ndim == 2:
            logging.warning(
                f"read audio {seg_id} with stereo channels, of shape {x.shape}"
            )
            x = np.sum(x, axis=0)  # sum channels if stereo
            assert len(x) > 10
            logging.warning("maximum/minimum values: %.3f/%.3f", x.max(), x.min())
        return x, fs[0]

    def _get_enable_codecs(self, seg_id):
        enable_codecs = {
            "enable_tel_codecs": True,
            "enable_media_codecs": True,
            "enable_transcodec": True,
        }
        if self.enable_tel_codecs_if is not None:
            enable_codecs["enable_tel_codecs"] = self.seg_set.loc[
                seg_id, "enable_tel_codecs"
            ]

        if self.enable_media_codecs_if is not None:
            enable_codecs["enable_media_codecs"] = self.seg_set.loc[
                seg_id, "enable_media_codecs"
            ]

        if self.enable_transcodec_if is not None:
            enable_codecs["enable_transcodec"] = self.seg_set.loc[
                seg_id, "enable_transcodec"
            ]

        return enable_codecs

    def _apply_aug_mix(self, x, x_augs, aug_idx):
        x_aug_mix = {}
        alpha_d = (self.aug_mix_alpha,) * len(x_augs)
        w = self.rng.dirichlet(alpha_d, self.num_aug_mix)
        m = self.rng.beta(alpha_d, self.num_aug_mix)
        for i in range(self.num_aug_mix):
            x_mix = np.zeros_like(x)
            for j, (_, x_aug_j) in enumerate(x_augs.items()):
                x_mix += w[i, j] * x_aug_j

            x_aug_mix[f"audio_aug_{aug_idx}_{i}"] = m[i] * x + (1 - m[i]) * x_mix

        return x_aug_mix

    def _apply_augs(
        self, x: np.array, duration: int, fs: int, enable_codecs: Dict[str, bool]
    ):
        if not self.augmenters:
            return {}

        if duration == 0:
            num_samples = len(x)
        else:
            num_samples = int(duration * fs)

        reverb_context_samples = len(x) - num_samples
        x_orig = x[reverb_context_samples:]
        x_augs = {}
        # for each type of augmentation
        for i, augmenter in enumerate(self.augmenters):
            # we do n_augs per augmentation type
            x_augs_i = {}
            for j in range(self.num_augs):
                # augment x
                x_aug, aug_attrs = augmenter(x, fs, **enable_codecs)
                # remove the extra left context used to compute the reverberation.
                x_aug = x_aug[reverb_context_samples : len(x)]
                x_aug = x_aug.astype(floatstr_torch(), copy=False)
                x_augs_i[f"audio_aug_{i}_{j}"] = x_aug

            if self.num_aug_mix > 0:
                x_augs_i = self._apply_aug_mix(x_orig, x_augs_i, i)

            x_augs.update(x_augs_i)

        if len(x_augs) == 1:
            # if we just have one aug so we just call it audio_aug
            x_augs["audio_aug"] = x_augs.pop("audio_aug_0_0")

        return x_augs

    def _get_segment_attrs(self, seg_id):
        seg_attrs = {}

        # tokenizer_name = ""
        # for attr_name, tokenizer_name in self.tokenizers_to_attrs.items():
        # if attr_name in self.tokenizers_to_attrs:
        #     tokenizer_name = info_name
        #     info_name = self.tokenizers_to_attrs[tokenizer_name]
        # elif tokenizer_name in self.tokenizers:
        #         seg_info_i = self.tokenizers[tokenizer_name].encode(seg_info_i)
        #     elif info_name == "text":
        #         seg_info_i = self.sp.encode(seg_info_i, out_type=int)

        for attr_name, tokenizer_name in self.output_attrs_to_tokenizers.items():
            in_attr_name = self.output_attrs_to_input[attr_name]
            in_attr = self.dataset.segments().loc[seg_id, in_attr_name]
            attr_tokens = self.tokenizers[tokenizer_name].encode(in_attr)
            seg_attrs[attr_name] = attr_tokens

        for class_name in self.class_names:
            class_id = self.dataset.segments().loc[seg_id, class_name]
            seg_attrs[class_name] = self.dataset.classes_value(class_name).loc[
                class_id, "class_idx"
            ]

        for attr_name in self.exta_attrs:
            seg_attrs[attr_name] = self.dataset.segments().loc[seg_id, attr_name]

        return seg_attrs

    def _resample(self, x, fs):
        if self.target_sample_freq is None:
            return x, fs

        return self.resampler(x, fs)

    def __getitem__(self, segment):
        self._maybe_reseed_worker()
        seg_id, start, duration = self._parse_segment_item(segment)
        audio, fs = self._read_audio(seg_id, start, duration)
        assert (
            len(audio) > 0
        ), f"read audio empty seg_id={seg_id}, start={start}, dur={duration}"
        audio, fs = self._resample(audio, fs)
        batch_data = {"id": seg_id, "sample_freq": fs, "audio": audio}
        enable_codecs = self._get_enable_codecs(seg_id)
        audio_augs = self._apply_augs(audio, duration, fs, enable_codecs)
        batch_data.update(audio_augs)
        seg_attrs = self._get_segment_attrs(seg_id)
        batch_data.update(seg_attrs)
        return batch_data

    @staticmethod
    def collate(batch):

        # sort batch by the length of audios
        audio_lengths = []
        for record in batch:
            audio_lengths.append(record["audio"].shape[0])

        audio_lengths = torch.as_tensor(audio_lengths)
        if not torch.all(audio_lengths[:-1] >= audio_lengths[1:]):
            sort_idx = torch.argsort(audio_lengths, descending=True)
            batch = [batch[i] for i in sort_idx]

        del audio_lengths

        def _is_list_of_tensors(x):
            return isinstance(x[0], (torch.Tensor, np.ndarray))

        def _is_list_of_items(x):
            return isinstance(
                x[0],
                (
                    int,
                    float,
                    np.int64,
                    np.int32,
                    np.int16,
                    np.int8,
                    np.float32,
                    np.float64,
                    np.float16,
                ),
            )

        def _is_list_of_strs(x):
            return isinstance(x[0], str)

        def _is_list_of_strlists(x):
            return isinstance(x[0], list) and isinstance(x[0][0], str)

        def _is_list_of_intlists(x):
            return isinstance(x[0], list) and isinstance(x[0][0], int)

        output_batch = {}
        batch_keys = batch[0].keys()
        for key in batch_keys:
            item_list = list_of_dicts_to_list(batch, key)
            if key == "id":
                # this are the segment ids
                output_batch[key] = item_list
            elif (
                key == "audio" or key[:6] == "audio_" and _is_list_of_tensors(item_list)
            ):
                # these are input audios
                assert item_list[0].ndim == 1, f"{batch[0]}"
                data, data_lengths = collate_seqs_1d(item_list)
                output_batch[key] = data
                output_batch[f"{key}_lengths"] = data_lengths
            elif _is_list_of_items(item_list):
                # these should be things like class ids
                output_batch[key] = torch.as_tensor(item_list)
            elif _is_list_of_tensors(item_list):
                # other tensor data
                data, data_lengths = collate_seqs_nd(item_list)
                output_batch[key] = data
                output_batch[f"{key}_lengths"] = data_lengths
            elif _is_list_of_intlists(item_list):
                # we assume k2 ragged tensor for now
                output_batch[key] = k2.RaggedTensor(item_list)
            elif _is_list_of_strs(item_list):
                # we just left them as they are:
                output_batch[key] = item_list
            else:
                raise TypeError(
                    f"we don't know how to collate {key} data={item_list} type={type(item_list[0])}"
                )

        return output_batch

    def get_collator(self):
        # return lambda batch: AudioDataset.collate(batch)
        return AudioDataset.collate

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(AudioDataset.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "dataset_path" not in skip:
            parser.add_argument(
                "--dataset-path",
                required=True,
                help="recordings manifest file (kaldi .scp or pandas .csv)",
            )

        parser.add_argument(
            "--class-names",
            default=None,
            nargs="+",
            help=(
                "list with the names of the types of classes that the dataset has to return, e.g., speaker, language"
            ),
        )

        parser.add_argument(
            "--tokenizer-mappings",
            default=None,
            nargs="+",
            help="""list mapping the segment_set fields to the tokenizer name 
            that should be used with them, e.g., text->text-1,
            this argument has to be sync with tokenizer_files.
            """,
        )

        parser.add_argument(
            "--tokenizer-files",
            default=None,
            nargs="+",
            help="""list of tokenizer cofinguration files
            this argument has to be sync with tokenizer_mappings.
            """,
        )

        parser.add_argument(
            "--bpe-model",
            default=None,
            help="bpe model for the text label",
        )

        if "aug_cfgs" not in skip:
            parser.add_argument(
                "--aug-cfgs",
                default=None,
                nargs="+",
                help="augmentation configuration file.",
            )

        parser.add_argument(
            "--num-augs",
            default=1,
            type=int,
            help="number of augmentations per segment and augmentation type",
        )
        parser.add_argument(
            "--num-aug-mix",
            default=0,
            type=int,
            help="number of AugMix augmentations per segment",
        )
        parser.add_argument(
            "--aug-mix-alpha",
            default=0.5,
            type=float,
            help="number of AugMix augmentations per segment",
        )

        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help=(
                "target sampling frequencey, if not None all audios are converted to this sample freq"
            ),
        )
        parser.add_argument(
            "--enable-tel-codecs-if",
            default=None,
            help="""condition to enable telephone codec augmentation, 
            for example use only if the segment is not conv. tel. speech: source_type != 'cts'""",
        )
        parser.add_argument(
            "--enable-media-codecs-if",
            default=None,
            help="""condition to enable media codec augmentation, 
            for example use only if the segment is audio from video: source_type == 'afv'""",
        )
        parser.add_argument(
            "--enable-transcodec-if",
            default=None,
            help="""condition to enable transcodec augmentation, 
            for example use transcodec only if the segment is spoof: spoof_det == 'spoof'""",
        )

        parser.add_argument(
            "--seed",
            default=11235811,
            type=int,
            help="random seed",
        )

        parser.add_argument(
            "--wav-scale",
            default=1.0,
            type=float,
            help=("multiplicative factor for waveform"),
        )

        parser.add_argument(
            "--target-sample-freq",
            default=None,
            type=int,
            help=("resample input audio to target frequency"),
        )
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))

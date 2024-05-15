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
from ...np.preprocessing import Resampler
from ...utils import ClassInfo, SegmentSet
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
      seed: random seed",
      time_durs_file: (deprecated) segment to duration in secs file, if durations are not in segments_file
      text_file: (deprecated) text file with words labels for each utterances.
      bpe_model: (deprecated) bpe model for the text label.
    """

    def __init__(
        self,
        recordings_file: str,
        segments_file: str,
        class_names: Optional[List[str]] = None,
        class_files: Optional[List[str]] = None,
        tokenizer_mappings: Optional[List[str]] = None,
        tokenizer_files: Optional[List[str]] = None,
        aug_cfgs: Optional[List[str]] = None,
        num_augs: int = 1,
        num_aug_mix: int = 0,
        aug_mix_alpha: float = 0,
        return_segment_info: Optional[List[str]] = None,
        return_orig: bool = False,
        target_sample_freq: Optional[float] = None,
        wav_scale: float = 1,
        is_val: bool = False,
        seed: int = 112358,
        time_durs_file: Optional[str] = None,
        text_file: Optional[str] = None,
        bpe_model: Optional[str] = None,
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
            logging.info("loading segments file %s", segments_file)

        self.seg_set = SegmentSet.load(segments_file)
        if rank == 0:
            logging.info("dataset contains %d seqs", len(self.seg_set))

        if rank == 0:
            logging.info("opening audio reader %s", recordings_file)

        audio_seg_set = self.seg_set if self.seg_set.has_time_marks else None
        self.r = AR(recordings_file, segments=audio_seg_set, wav_scale=wav_scale)

        self.is_val = is_val
        if time_durs_file is not None:
            self._load_legacy_durations(time_durs_file)

        assert "duration" in self.seg_set

        logging.info("loading class-info files")
        self._load_class_infos(class_names, class_files, is_val)

        logging.info("loading tokenizers")
        self._load_tokenizers(tokenizer_mappings, tokenizer_files)

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
        self.num_aug_mix = num_aug_mix
        self.aug_mix_alpha = aug_mix_alpha
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000 * rank)
        self._create_augmenters(aug_cfgs)

        self.target_sample_freq = target_sample_freq
        self.resamplers = {}
        self.resampler = Resampler(target_sample_freq)

    def _load_legacy_durations(self, time_durs_file):
        if self.rank == 0:
            logging.info("loading durations file %s", time_durs_file)

        time_durs = SegmentSet.load(time_durs_file)
        self.seg_set["duration"] = time_durs.loc[
            self.seg_set["id"]
        ].class_id.values.astype(np.float, copy=False)

    def _load_bpe_model(self, bpe_model, is_val):
        if self.rank == 0:
            logging.info("loading bpe file %s", bpe_model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        blank_id = self.sp.piece_to_id("<blk>")
        vocab_size = self.sp.get_piece_size()

    def _load_text_infos(self, text_file, is_val):
        if text_file is None:
            return
        if self.rank == 0:
            logging.info("loading text file %s", text_file)

        text = read_text(text_file)
        self.seg_set["text"] = text.loc[self.seg_set["id"]].text

    def _load_class_infos(self, class_names, class_files, is_val):
        self.class_info = OrderedDict()
        if class_names is None:
            assert class_files is None
            return

        assert len(class_names) == len(class_files)
        for name, file in zip(class_names, class_files):
            assert (
                name in self.seg_set
            ), f"class_name {name} not present in the segment set"
            if self.rank == 0:
                logging.info("loading class-info file %s", file)
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

    def _load_tokenizers(self, tokenizer_mappings, tokenizer_files):
        self.tokenizers = OrderedDict()
        self.tokenizers_to_infos = OrderedDict()
        if tokenizer_mappings is None:
            assert tokenizer_files is None
            return

        assert len(tokenizer_mappings) == len(tokenizer_files)
        tokenizer_names = []
        for map in tokenizer_mappings:
            info_name, tokenizer_name = map.split("->", maxsplit=1)
            self.tokenizers_to_infos[tokenizer_name] = info_name
            tokenizer_names.append(tokenizer_name)

        for name, file in zip(tokenizer_names, tokenizer_files):
            assert name in self.seg_set, f"field {name} not present in the segment set"
            if self.rank == 0:
                logging.info("loading tokenizer file %s", file)
            tokenizer = HypTokenizer.auto_load(file)
            self.tokenizers[name] = tokenizer

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

        return seg_id, start, duration

    def _read_audio(self, seg_id, start, duration):
        # how much extra audio we need to load to
        # calculate the reverb of the first part of the audio
        reverb_context = min(self.reverb_context, start)
        start -= reverb_context
        read_duration = duration + reverb_context

        # read audio
        x, fs = self.r.read([seg_id], time_offset=start, time_durs=read_duration)
        return x[0].astype(floatstr_torch(), copy=False), fs[0]

    def _apply_aug_mix(self, x, x_augs, aug_idx):
        x_aug_mix = {}
        alpha_d = (self.aug_mix_alpha,) * len(x_augs)
        w = self.rng.dirichlet(alpha_d, self.num_aug_mix)
        m = self.rng.beta(alpha_d, self.num_aug_mix)
        for i in range(self.num_aug_mix):
            x_mix = np.zeros_like(x)
            for j, (_, x_aug_j) in enumerate(x_augs.items()):
                x_mix += w[i, j] * x_aug_j

            x_aug_mix[f"x_aug_{aug_idx}_{i}"] = m[i] * x + (1 - m[i]) * x_mix

        return x_aug_mix

    def _apply_augs(self, x, duration, fs):
        if not self.augmenters:
            return {"x": x}

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
                x_aug, aug_info = augmenter(x)
                # remove the extra left context used to compute the reverberation.
                x_aug = x_aug[reverb_context_samples : len(x)]
                x_aug = x_aug.astype(floatstr_torch(), copy=False)
                x_augs_i[f"x_aug_{i}_{j}"] = x_aug

            if self.num_aug_mix > 0:
                x_augs_i = self._apply_aug_mix(x_orig, x_augs_i, i)

            x_augs.update(x_augs_i)

        if self.return_orig:
            x_augs["x"] = x_orig
        elif len(x_augs) == 1:
            # if we just have one aug and we don't return the clean version,
            # we just call x to the aug version
            x_augs["x"] = x_augs.pop("x_aug_0_0")

        return x_augs

    def _get_segment_info(self, seg_id):
        seg_info = {}
        # converts the class_ids to integers
        for info_name in self.return_segment_info:
            tokenizer_name = ""
            if info_name in self.tokenizers_to_infos:
                tokenizer_name = info_name
                info_name = self.tokenizers_to_infos[tokenizer_name]

            seg_info_i = self.seg_set.loc[seg_id, info_name]
            if info_name in self.class_info:
                # if the type of information is a class-id
                # we use the class information table to
                # convert from id to integer
                class_info = self.class_info[info_name]
                seg_info_i = class_info.loc[seg_info_i, "class_idx"]
            elif tokenizer_name in self.tokenizers:
                seg_info_i = self.tokenizers[tokenizer_name].encode(seg_info_i)
            elif info_name == "text":
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
        if self.target_sample_freq is None:
            return x, fs

        return self.resampler(x, fs)

    def __getitem__(self, segment):
        seg_id, start, duration = self._parse_segment_item(segment)
        x, fs = self._read_audio(seg_id, start, duration)
        assert (
            len(x) > 0
        ), f"read audio empty seg_id={seg_id}, start={start}, dur={duration}"
        x, fs = self._resample(x, fs)
        data = {"seg_id": seg_id, "sample_freq": fs}
        x_augs = self._apply_augs(x, duration, fs)
        data.update(x_augs)
        seg_info = self._get_segment_info(seg_id)
        data.update(seg_info)
        return data

    @staticmethod
    def collate(self, batch):

        # sort batch by the length of x
        audio_lengths = []
        for record in batch:
            audio_lengths.append(record["x"].shape[0])
        audio_lengths = torch.as_tensor(audio_lengths)
        if not torch.all(audio_lengths[:-1] >= audio_lengths[1:]):
            sort_idx = torch.argsort(audio_lengths, descending=True)
            batch = [batch[i] for i in sort_idx]

        del audio_lengths

        def _is_list_of_tensors(x):
            return isinstance(x[0], (torch.Tensor, np.ndarray))

        def _is_list_of_items(x):
            return isinstance(x[0], (int, float))

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
            elif key == "x" or key[:2] == "x_" and _is_list_of_tensors(item_list):
                # these are input audios
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
                raise TypeError(f"we don't know how to collate this data={item_list}")

        return output_batch

    @staticmethod
    def collate_old(self, batch):
        from torch.nn.utils.rnn import pad_sequence

        audio = []
        audio_length = []
        target = []
        for record in batch:
            audio_length.append(record["x"].shape[0])
        audio_length = torch.as_tensor(audio_length)
        if not torch.all(audio_length[:-1] >= audio_length[1:]):
            sort_idx = torch.argsort(audio_length, descending=True)
            batch = [batch[i] for i in sort_idx]

        audio_length = []
        for record in batch:
            wav = torch.as_tensor(record["x"])
            audio.append(wav)
            audio_length.append(wav.shape[0])
            target.append(record["text"])

        audio = pad_sequence(audio)
        audio_length = torch.as_tensor(audio_length)
        target = k2.RaggedTensor(target)
        batch = {
            "x": torch.transpose(audio, 0, 1),
            "x_lengths": audio_length,
            "text": target,
        }
        return batch

    def get_collator(self):
        return lambda batch: AudioDataset.collate(self, batch)

    @staticmethod
    def filter_args(**kwargs):
        args = filter_func_args(AudioDataset.__init__, kwargs)
        return args

    @staticmethod
    def add_class_args(parser, prefix=None, skip=set()):
        if prefix is not None:
            outer_parser = parser
            parser = ArgumentParser(prog="")

        if "recordings_file" not in skip:
            parser.add_argument(
                "--recordings-file",
                required=True,
                help="recordings manifest file (kaldi .scp or pandas .csv)",
            )

        if "segments_file" not in skip:
            parser.add_argument(
                "--segments-file",
                required=True,
                help="segments manifest file (kaldi .scp or pandas .csv)",
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
            "--class-files",
            default=None,
            nargs="+",
            help="list of class info files",
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
            "--time-durs-file",
            default=None,
            help=(
                "(deprecated) segment to duration in secs file, if durations are not in segments_file"
            ),
        )

        parser.add_argument(
            "--bpe-model",
            default=None,
            help="bpe model for the text label",
        )

        parser.add_argument(
            "--text-file",
            default=None,
            help="text file with words labels for each utterances",
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

        parser.add_argument(
            "--seed",
            default=11235811,
            type=int,
            help="random seed",
        )

        AR.add_class_args(parser)
        if prefix is not None:
            outer_parser.add_argument("--" + prefix, action=ActionParser(parser=parser))
            # help='audio dataset options')

    add_argparse_args = add_class_args

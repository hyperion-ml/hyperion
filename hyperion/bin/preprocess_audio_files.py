#!/usr/bin/env python
"""
Copyright 2020 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from scipy import ndimage

from hyperion.hyp_defs import config_logger
from hyperion.io import AudioWriter as Writer
from hyperion.io import SequentialAudioReader as AR
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.preprocessing import ResamplerToTargetFreq
from hyperion.utils import HyperDataset, PathLike, Utt2Info


def _infer_table_sep(dataset: Optional[HyperDataset]) -> str:
    if dataset is None:
        return ","
    table_sep = dataset.table_sep
    if table_sep is None:
        return ","
    return table_sep


def resample_vad(vad: np.ndarray, length: int) -> np.ndarray:
    """Resample a VAD boolean array to a target length using nearest indices.

    Args:
        vad: VAD mask array.
        length: Target number of samples.
    """
    if length <= 0:
        return vad[:0]
    step = (len(vad) - 1) / length
    assert step < 1
    idx = step * np.arange(length, dtype=float)
    idx = np.round(idx).astype(int)
    return vad[idx]


def process_vad(
    vad: np.ndarray, length: int, fs: float, dilation: float, erosion: float
) -> np.ndarray:
    """Resample and optionally dilate/erode the VAD mask to match audio length.

    Args:
        vad: VAD mask array.
        length: Target number of samples.
        fs: Audio sampling frequency in Hz.
        dilation: Dilation in seconds.
        erosion: Erosion in seconds.
    """
    # vad = signal.resample(vad, length) > 0.5
    vad = resample_vad(vad, length)
    if dilation > 0:
        iters = int(dilation * fs)
        vad = ndimage.binary_dilation(vad, iterations=iters)

    if erosion > 0:
        iters = int(erosion * fs)
        vad = ndimage.binary_erosion(vad, iterations=iters, border_value=True)

    return vad


def process_audio_files(
    dataset: Optional[PathLike],
    segments_file: Optional[PathLike],
    recordings_file: Optional[PathLike],
    output_path: PathLike,
    output_dataset: Optional[PathLike],
    output_segments_file: Optional[PathLike],
    output_recordings_file: Optional[PathLike],
    write_time_durs_spec: Optional[PathLike],
    vad_spec: Optional[PathLike],
    vad_path_prefix: Optional[PathLike] = None,
    vad_fs: float = 100,
    vad_dilation: float = 0,
    vad_erosion: float = 0,
    remove_dc_offset: bool = False,
    target_sample_freq: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """Read audio from dataset, optionally apply VAD/resampling, and write outputs.

    Args:
        dataset: Optional Hyperion dataset spec.
        segments_file: Optional Hyperion segments file.
        recordings_file: Optional Hyperion recordings file.
        output_path: Output directory or base path for audio files.
        output_dataset: Optional output dataset file.
        output_segments_file: Optional output segments file.
        output_recordings_file: Optional output recordings file.
        write_time_durs_spec: Optional path to write utterance durations.
        vad_spec: Optional VAD spec for trimming to speech.
        vad_path_prefix: Optional prefix for VAD file paths.
        vad_fs: VAD frame rate in Hz.
        vad_dilation: Dilation in seconds.
        vad_erosion: Erosion in seconds.
        remove_dc_offset: Whether to remove DC offset.
        target_sample_freq: Optional target sample rate in Hz.
        **kwargs: Additional reader/writer keyword arguments.
    """
    input_args = AR.filter_args(**kwargs)
    output_args = Writer.filter_args(**kwargs)
    logging.info(f"input_args={input_args}")
    logging.info(f"output_args={output_args}")
    if output_recordings_file is None and output_dataset is None:
        raise ValueError("output_recordings_file or output_dataset is required")
    if target_sample_freq is not None:
        resampler = ResamplerToTargetFreq(target_sample_freq)
    else:
        resampler = None

    if write_time_durs_spec is not None:
        keys = []
        info = []

    if output_dataset is not None and (
        output_recordings_file is None or output_segments_file is None
    ):
        dataset_dir, _ = HyperDataset.resolve_dataset_path(output_dataset)
        table_sep = _infer_table_sep(
            HyperDataset.load(dataset) if dataset is not None else None
        )
        table_ext = ".tsv" if table_sep == "\t" else ".csv"
        if output_recordings_file is None:
            output_recordings_file = dataset_dir / f"recordings{table_ext}"
        if output_segments_file is None:
            output_segments_file = dataset_dir / f"segments{table_ext}"

    output_segments = None

    with (
        AR(
            dataset=dataset,
            segments=segments_file,
            recordings=recordings_file,
            **input_args,
        ) as reader,
        Writer(output_path, output_recordings_file, **output_args) as writer,
    ):
        if vad_spec is not None:
            logging.info("opening VAD stream: %s", vad_spec)
            v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

        t1 = time.time()
        for data in reader:
            key, x, fs = data
            logging.info("Processing audio %s", key)
            t2 = time.time()

            if resampler is not None:
                x, fs = resampler(x, fs)

            tot_samples = x.shape[0]
            if vad_spec is not None:
                num_vad_frames = int(round(tot_samples * vad_fs / fs))
                if num_vad_frames <= 0:
                    logging.warning(
                        "utt %s has non-positive number of VAD frames %d",
                        key,
                        num_vad_frames,
                    )
                else:
                    logging.info(
                        "utt %s tot-samples=%d fs=%f num-vad-frames=%d",
                        key,
                        tot_samples,
                        fs,
                        num_vad_frames,
                    )
                    vad = v_reader.read(key, num_frames=num_vad_frames)[0].astype(
                        "bool", copy=False
                    )
                    logging.info("vad=%d/%d", np.sum(vad == 1), len(vad))
                    vad = process_vad(vad, tot_samples, fs, vad_dilation, vad_erosion)
                    logging.info("vad=%d/%d", np.sum(vad == 1), len(vad))
                    x = x[vad]

            logging.info(
                "utt %s detected %f/%f secs (%.2f %%) speech ",
                key,
                x.shape[0] / fs,
                tot_samples / fs,
                x.shape[0] / tot_samples * 100,
            )

            if x.shape[0] > 0:
                if remove_dc_offset:
                    x -= np.mean(x)

                writer.write([key], [x], [fs])
                if write_time_durs_spec is not None:
                    keys.append(key)
                    info.append(x.shape[0] / fs)

                xmax = np.max(x)
                xmin = np.min(x)
            else:
                xmax = 0
                xmin = 0

            t3 = time.time()
            dt2 = (t2 - t1) * 1000
            dt3 = (t3 - t1) * 1000
            time_dur = len(x) / fs
            rtf = (time_dur * 1000) / dt3
            logging.info(
                (
                    "Packed audio %s length=%0.3f secs "
                    "elapsed-time=%.2f ms. "
                    "read-time=%.2f ms. write-time=%.2f ms. "
                    "real-time-factor=%.2f "
                    "x-range=[%f - %f]"
                ),
                key,
                time_dur,
                dt3,
                dt2,
                dt3 - dt2,
                rtf,
                xmin,
                xmax,
            )
            t1 = time.time()
            if output_segments is None and reader.with_segments:
                output_segments = reader.segments.copy()
                if "recording" in output_segments.columns:
                    output_segments.drop(columns=["recording"], inplace=True)
                if "start" in output_segments.columns:
                    output_segments.drop(columns=["start"], inplace=True)

            if output_segments is not None and vad_spec is not None:
                output_segments.loc[key, "duration"] = time_dur

    if write_time_durs_spec is not None:
        logging.info("writing time durations to %s", write_time_durs_spec)
        u2td = Utt2Info.create(keys, info)
        u2td.save(write_time_durs_spec)

    if output_segments_file is not None and output_segments is not None:
        logging.info("writing segments to %s", output_segments_file)
        output_segments.save(output_segments_file)

    if output_dataset is not None:
        logging.info("writing dataset to %s", output_dataset)
        if dataset is not None:
            dataset = HyperDataset.load(dataset)
            dataset.set_recordings(output_recordings_file)
            dataset.set_segments(output_segments_file)
        else:
            dataset = HyperDataset(
                recordings=output_recordings_file,
                segments=output_segments_file,
            )
        dataset.save(output_dataset)


def main() -> None:
    """Parse CLI arguments and run audio preprocessing.

    Args:
        None.
    """
    parser = ArgumentParser(
        description=(
            "Process audio entries from a Hyperion dataset or recordings table, "
            "optionally apply VAD and resampling, and write outputs."
        )
    )

    parser.add_argument(
        "--cfg",
        action=ActionConfigFile,
        help="Configuration file in jsonargparse format.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Input Hyperion dataset.",
    )
    parser.add_argument(
        "--segments-file",
        default=None,
        help=(
            "Optional Hyperion segments file. Used only when --recordings-file is "
            "provided; ignored when --dataset is provided."
        ),
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help=(
            "Hyperion recordings file. Required when --dataset is not provided; "
            "ignored when --dataset is provided."
        ),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory or base path where audio files will be written.",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help=(
            "Optional Hyperion output dataset file. When set, defaults "
            "--output-recordings-file/--output-segments-file based on table separator."
        ),
    )
    parser.add_argument(
        "--output-segments-file",
        default=None,
        help=(
            "Optional Hyperion output segments file. If omitted and --output-dataset "
            "is set, defaults to output_dataset/segments.csv or .tsv."
        ),
    )
    parser.add_argument(
        "--output-recordings-file",
        default=None,
        help=(
            "Optional Hyperion output recordings file. Required when --output-dataset "
            "is not provided; defaults to output_dataset/recordings.csv or .tsv."
        ),
    )
    parser.add_argument(
        "--write-time-durs",
        dest="write_time_durs_spec",
        default=None,
        help="Write per-utterance durations (seconds) to this file.",
    )
    parser.add_argument(
        "--vad",
        dest="vad_spec",
        default=None,
        help="VAD source spec (e.g., scp/ark). If set, trims audio to speech.",
    )
    parser.add_argument(
        "--vad-path-prefix",
        default=None,
        help="Prefix to prepend to VAD file paths in scp files.",
    )

    parser.add_argument(
        "--vad-fs",
        default=100,
        type=float,
        help="VAD frame rate in Hz.",
    )

    parser.add_argument(
        "--vad-dilation",
        default=0,
        type=float,
        help="Dilate VAD mask by this many seconds.",
    )

    parser.add_argument(
        "--vad-erosion",
        default=0,
        type=float,
        help="Erode VAD mask by this many seconds (after dilation).",
    )

    AR.add_class_args(parser)
    Writer.add_class_args(parser)
    parser.add_argument(
        "--remove-dc-offset",
        default=False,
        action="store_true",
        help="Remove DC offset from audio before writing.",
    )
    parser.add_argument(
        "--target-sample-freq",
        default=None,
        type=float,
        help="Resample audio to this sample rate (Hz).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbosity level (0=errors, 3=debug).",
    )
    args = parser.parse_args()
    if args.output_recordings_file is None and args.output_dataset is None:
        parser.error("--output-recordings-file or --output-dataset is required")
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    process_audio_files(**namespace_to_dict(args))


if __name__ == "__main__":
    main()

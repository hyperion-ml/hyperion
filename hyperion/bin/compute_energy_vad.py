#!/usr/bin/env python
"""
Copyright 2018 Jesus Villalba (Johns Hopkins University)
Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import os
import sys
import time
from pathlib import Path

# --- plotting (headless) ---
import matplotlib

matplotlib.use("Agg")  # ensure no GUI needed
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from scipy import ndimage

from hyperion.hyp_defs import config_logger
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialAudioReader as AR
from hyperion.np.feats import EnergyVAD, st_logE
from hyperion.np.preprocessing import ResamplerToTargetFreq


def save_vad_2channel_plot(
    output_dir,
    key,  # string to include in filename
    channel,  # int: channel index of target channel
    x,  # 1D np.ndarray (target channel, padded if needed)
    x_side,  # 1D np.ndarray (side channel, padded if needed)
    y_init,  # 1D bool array: initial VAD on target (y)
    y_side_init,  # 1D bool array: initial VAD on side (y_side before energy compare)
    y_side_after,  # 1D bool array: side VAD after energy comparison
    y_tgt,  # 1D bool array: final target VAD after removing side
    fs,  # sample rate (int)
    frame_shift,  # frame shift in ms
):
    """
    Save a PNG showing waveforms for target/side channels and step overlays for:
      - target VAD (initial and final)
      - side VAD (initial and after energy-based assignment)

    Returns: Path to the saved PNG.
    """
    # --- checks ---
    n = len(y_init)
    assert (
        y_side_init.shape[0] == n and y_side_after.shape[0] == n
    ), "VAD lengths must match"
    assert y_tgt.shape[0] == n, "y_tgt length must match y_init"

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{key}_vad.png"
    out_path = out_dir / fname

    # time axes
    t_audio = np.arange(len(x)) / fs
    t_audio_side = np.arange(len(x_side)) / fs
    t_frames = (np.arange(n + 1) * frame_shift) / 1000  # steps-post needs +1

    # scale VAD to waveform amplitude for visible overlay
    def _scale(v, sig):
        amp = 0.9 * (np.max(np.abs(sig)) + 1e-8)
        v_step = np.concatenate([v.astype(float), [v[-1]]])  # extend for steps-post
        return v_step * amp

    y_init_scaled = 0.9 * _scale(y_init, x)
    y_side_init_scaled = 0.9 * _scale(y_side_init, x_side)
    y_side_after_scaled = _scale(y_side_after, x_side)
    y_tgt_scaled = _scale(y_tgt, x)
    channel = channel + 1  # for display only
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # target channel
    axs[0].plot(t_audio, x, lw=0.6, label="wave (target)")
    axs[0].plot(
        t_frames,
        y_init_scaled,
        drawstyle="steps-post",
        lw=1.2,
        alpha=0.8,
        color="red",
        label="VAD target (init)",
    )
    axs[0].plot(
        t_frames,
        y_tgt_scaled,
        drawstyle="steps-post",
        lw=1.2,
        color="green",
        label="VAD target (final)",
    )
    axs[0].set_ylabel("amplitude")
    axs[0].set_title(
        f"{key} — target channel={channel}" if key else f"Target channel={channel}"
    )
    axs[0].grid(True, alpha=0.2)
    axs[0].legend(loc="upper right")

    # side channel
    axs[1].plot(t_audio_side, x_side, lw=0.6, label="wave (side)")
    axs[1].plot(
        t_frames,
        y_side_init_scaled,
        drawstyle="steps-post",
        lw=1.2,
        alpha=0.8,
        color="red",
        label="VAD side (init)",
    )
    axs[1].plot(
        t_frames,
        y_side_after_scaled,
        drawstyle="steps-post",
        lw=1.2,
        color="green",
        label="VAD side (after energy)",
    )
    axs[1].set_ylabel("amplitude")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_title(f"{key} — side" if key else "Side")
    axs[1].grid(True, alpha=0.2)
    axs[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def save_vad_1channel_plot(
    output_dir,
    key,  # string to include in filename
    x,  # 1D np.ndarray (target channel, padded if needed)
    y,  # 1D bool array: final target VAD
    fs,  # sample rate (int)
    frame_shift,  # frame shift in ms
):
    """
    Save a PNG showing one-channel waveform with VAD overlay.

    Args:
        output_dir: str | Path, output directory.
        key: str, used in filename.
        x: 1D array-like, waveform samples.
        y: 1D bool array, VAD per frame.
        fs: int, sample rate (Hz).
        frame_shift_ms: float, VAD frame shift in milliseconds.
        y_color: str, color for VAD overlay line.

    Returns:
        Path to the saved PNG.
    """
    # --- checks ---
    n = len(y)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{key}_vad.png"
    out_path = out_dir / fname

    # time axes
    t_audio = np.arange(len(x)) / fs
    t_frames = (np.arange(n + 1) * frame_shift) / 1000  # steps-post needs +1

    # scale VAD to waveform amplitude for visible overlay
    def _scale(v, sig):
        amp = 0.9 * (np.max(np.abs(sig)) + 1e-8)
        v_step = np.concatenate([v.astype(float), [v[-1]]])  # extend for steps-post
        return v_step * amp

    y_tgt_scaled = _scale(y, x)
    fig, axs = plt.subplots(1, 1, figsize=(12, 6))

    # target channel
    axs.plot(t_audio, x, lw=0.6, label="wave")
    axs.plot(
        t_frames,
        y_tgt_scaled,
        drawstyle="steps-post",
        lw=1.2,
        color="green",
        label="VAD",
    )
    axs.set_ylabel("amplitude")
    if key:
        axs.set_title(f"{key}")
    axs.grid(True, alpha=0.2)
    axs.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def compute_vad(
    dataset_file,
    recordings_file,
    segments_file,
    output_spec,
    write_num_frames,
    remove_cross_talk,
    min_post_cross_talk_dur,
    vad_plot_dir,
    **kwargs,
):
    vad_args = EnergyVAD.filter_args(**kwargs)
    vad = EnergyVAD(**vad_args)
    resampler = ResamplerToTargetFreq(vad.sample_frequency)

    input_args = AR.filter_args(**kwargs)
    reader = AR(
        dataset=dataset_file,
        recordings=recordings_file,
        segments=segments_file,
        target_sample_freq=vad.sample_frequency,
        return_all_channels=True if remove_cross_talk else False,
        **input_args,
    )

    metadata_columns = [
        "frame_shift",
        "frame_length",
        "num_frames",
        "num_speech_frames",
        "prob_speech",
    ]

    writer = DWF.create(output_spec, metadata_columns=metadata_columns)

    if write_num_frames is not None:
        f_num_frames = open(write_num_frames, "w")

    for data in reader:
        if remove_cross_talk:
            key, x, fs, channel = data
        else:
            key, x, fs = data

        logging.info("Extracting VAD for %s", key)
        if fs != vad.sample_frequency:
            logging.info("Resampling from %d Hz to %d Hz", fs, vad.sample_frequency)
            x, fs = resampler(x, fs)

        if remove_cross_talk:
            num_channels = x.shape[0]
            if num_channels > 1:
                x_side = x[1] if channel == 0 else x[0]
                x = x[channel]
            else:
                x = x[0]

        x_dur = len(x) / fs
        t1 = time.time()
        y = vad.compute(x)
        if remove_cross_talk and num_channels > 1:
            # zero out the speech frames where cross-talk is detected
            y_side = vad.compute(x_side)

            frame_length = int(vad.frame_length * vad.sample_frequency / 1000)
            frame_shift = int(vad.frame_shift * vad.sample_frequency / 1000)
            tgt_x_len = (y.shape[0] - 1) * frame_shift + frame_length
            delta = tgt_x_len - len(x)
            left_pad = delta // 2
            right_pad = delta - left_pad
            if left_pad > 0 or right_pad > 0:
                x = np.pad(x, (left_pad, right_pad), mode="constant")
                x_side = np.pad(x_side, (left_pad, right_pad), mode="constant")

            log_energy = st_logE(x, frame_length, frame_shift)
            log_energy_side = st_logE(x_side, frame_length, frame_shift)
            assign_to_side = log_energy_side > log_energy
            y_side_after = np.logical_and(y_side, assign_to_side)
            y_tgt = np.logical_and(y, np.logical_not(y_side_after))
            y_tgt = ndimage.binary_dilation(y_tgt, iterations=8)
            y_tgt = ndimage.binary_erosion(y_tgt, iterations=4, border_value=True)
            if vad_plot_dir is not None:
                plot_path = save_vad_2channel_plot(
                    vad_plot_dir,
                    key,
                    channel,
                    x,
                    x_side,
                    y,
                    y_side,
                    y_side_after,
                    y_tgt,
                    fs,
                    vad.frame_shift,
                )
                logging.info("Saved VAD debug plot to %s", plot_path)
            overlap_ratio = np.sum(np.logical_and(y, y_side)) / np.sum(y)
            logging.info(
                "Cross-talk removal %s, overlap ratio=%f %%", key, overlap_ratio * 100
            )
            speech_dur = np.sum(y_tgt) * vad.frame_shift / 1000  # frame shift in ms
            if speech_dur > min_post_cross_talk_dur:
                y = y_tgt
            else:
                logging.info(
                    "Post cross-talk removal speech duration %f s is less than %f s, we keep the original VAD",
                    speech_dur,
                    min_post_cross_talk_dur,
                )
        else:
            if vad_plot_dir is not None:
                print("here", x.shape, y.shape, flush=True)
                plot_path = save_vad_1channel_plot(
                    vad_plot_dir, key, x, y, fs, vad.frame_shift
                )
                print("xxx", flush=True)
                logging.info("Saved VAD debug plot to %s", plot_path)

        dt = (time.time() - t1) * 1000
        rtf = vad.frame_shift * y.shape[0] / dt
        num_speech_frames = np.sum(y)
        prob_speech = num_speech_frames / y.shape[0] * 100

        logging.info(
            "Extracted VAD for %s detected %d/%d (%f %%) speech frames, elapsed-time=%.2f ms. real-time-factor=%.2f",
            key,
            num_speech_frames,
            y.shape[0],
            prob_speech,
            dt,
            rtf,
        )
        metadata = {
            "frame_shift": vad.frame_shift,
            "frame_length": vad.frame_length,
            "num_frames": y.shape[0],
            "num_speech_frames": num_speech_frames,
            "prob_speech": prob_speech,
            "speech_duration": num_speech_frames * vad.frame_shift / 1000,
            "duration": x_dur,
        }
        writer.write([key], [y], metadata)
        if write_num_frames is not None:
            f_num_frames.write("%s %d\n" % (key, y.shape[0]))

        vad.reset()

    if write_num_frames is not None:
        f_num_frames.close()


def main():
    parser = ArgumentParser(description="Compute Kaldi Energy VAD")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--dataset-file", default=None)
    parser.add_argument("--recordings-file", default=None)
    parser.add_argument("--segments-file", default=None)
    parser.add_argument("--output-spec", required=True)
    parser.add_argument("--write-num-frames", default=None)
    parser.add_argument("--write-stats", default=None)
    parser.add_argument("--remove-cross-talk", default=False, action=ActionYesNo)
    parser.add_argument("--min-post-cross-talk-dur", type=float, default=10.0)
    parser.add_argument(
        "--vad-plot-dir",
        type=str,
        default=None,
        help="Directory to save VAD debug plots",
    )

    AR.add_class_args(parser)
    EnergyVAD.add_class_args(parser)
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbose level",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    compute_vad(**namespace_to_dict(args))


if __name__ == "__main__":
    main()

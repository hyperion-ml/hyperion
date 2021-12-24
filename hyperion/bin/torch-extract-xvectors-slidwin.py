#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import sys
import os
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)
import time
import logging

import numpy as np

import torch

from hyperion.hyp_defs import config_logger, float_cpu
from hyperion.utils import Utt2Info
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.feats import MeanVarianceNorm as MVN

from hyperion.torch.utils import open_device
from hyperion.torch import TorchModelLoader as TML


def init_device(use_gpu):
    set_float_cpu("float32")
    num_gpus = 1 if use_gpu else 0
    logging.info("initializing devices num_gpus={}".format(num_gpus))
    device = open_device(num_gpus=num_gpus)
    return device


def init_mvn(device, **kwargs):
    mvn_args = MVN.filter_args(**kwargs["mvn"])
    logging.info("mvn args={}".format(mvn_args))
    mvn = MVN(**mvn_args)
    if mvn.norm_mean or mvn.norm_var:
        return mvn
    return None


def load_model(model_path, device):
    logging.info("loading model {}".format(model_path))
    model = TML.load(model_path)
    logging.info("xvector-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def extract_xvectors(
    input_spec,
    output_spec,
    vad_spec,
    write_timestamps_spec,
    slidwin_params_path,
    vad_path_prefix,
    model_path,
    chunk_length,
    embed_layer,
    win_length,
    win_shift,
    snip_edges,
    feat_frame_length,
    feat_frame_shift,
    feat_snip_edges,
    use_gpu,
    **kwargs
):

    logging.info("initializing")
    rng = np.random.RandomState(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    mvn = init_mvn(device, **kwargs)
    model = load_model(model_path, device)

    if write_timestamps_spec is not None:
        time_writer = DWF.create(write_timestamps_spec, scp_sep=scp_sep)

    dr_args = DRF.filter_args(**kwargs)
    logging.info("opening output stream: %s" % (output_spec))
    with DWF.create(output_spec) as writer:

        logging.info("opening input stream: %s" % (output_spec))
        with DRF.create(input_spec, **dr_args) as reader:
            if vad_spec is not None:
                logging.info("opening VAD stream: %s" % (vad_spec))
                v_reader = VRF.create(vad_spec, path_prefix=vad_path_prefix)

            while not reader.eof():
                t1 = time.time()
                key, data = reader.read(1)
                if len(key) == 0:
                    break
                t2 = time.time()
                logging.info("processing utt %s" % (key[0]))
                x = data[0]
                if mvn is not None:
                    x = mvn.normalize(x)
                t3 = time.time()
                tot_frames = x.shape[0]
                if vad_spec is not None:
                    vad = v_reader.read(key, num_frames=x.shape[0])[0].astype(
                        "bool", copy=False
                    )
                    x = x[vad]

                logging.info(
                    "utt %s detected %d/%d (%.2f %%) speech frames"
                    % (key[0], x.shape[0], tot_frames, x.shape[0] / tot_frames * 100)
                )

                t4 = time.time()
                if x.shape[0] == 0:
                    y = np.zeros(
                        (
                            1,
                            model.embed_dim,
                        ),
                        dtype=float_cpu(),
                    )
                else:
                    xx = torch.tensor(x.T[None, :], dtype=torch.get_default_dtype())
                    with torch.no_grad():
                        y = (
                            model.extract_embed_slidwin(
                                xx,
                                win_length,
                                win_shift,
                                snip_edges=snip_edges,
                                feat_frame_length=feat_frame_length,
                                feat_frame_shift=feat_frame_shift,
                                chunk_length=chunk_length,
                                embed_layer=embed_layer,
                                detach_chunks=True,
                            )
                            .detach()
                            .cpu()
                            .numpy()[0]
                        )

                        # if np.any(np.isnan(y)):
                        #     y = y.T
                        #     idx = np.any(np.isnan(y), axis=1)
                        #     print(y[idx])
                        #     print(np.where(idx))
                        #     raise Exception()
                        # y1 = model.extract_embed(
                        #     xx[:,:,:148],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y.shape, y1.shape))
                        # logging.info('{} {}'.format(y[:20, 0], y1[:20]))
                        # y2 = model.extract_embed(
                        #     xx[:,:,25:173],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 1], y2[:20]))
                        # y3 = model.extract_embed(
                        #      xx[:,:,250:398],
                        #      chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 10], y3[:20]))

                        # win_length = 20
                        # y = model.extract_embed_slidwin(
                        #     xx, win_length, win_shift, snip_edges=True,
                        #     feat_frame_length=feat_frame_length, feat_frame_shift=feat_frame_shift,
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]

                        # y1 = model.extract_embed(
                        #     xx[:,:,:1999],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y.shape, y1.shape))
                        # logging.info('{} {}'.format(y[:20, 0], y1[:20]))
                        # y2 = model.extract_embed(
                        #     xx[:,:,25:2024],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 1], y2[:20]))
                        # y3 = model.extract_embed(
                        #      xx[:,:,250:2249],
                        #      chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 10], y3[:20]))

                        # win_length = 20
                        # y = model.extract_embed_slidwin(
                        #     xx, win_length, win_shift, snip_edges=False,
                        #     feat_frame_length=feat_frame_length, feat_frame_shift=feat_frame_shift,
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]

                        # y1 = model.extract_embed(
                        #     xx[:,:,:1112],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y.shape, y1.shape))
                        # logging.info('{} {}'.format(y[:20, 0], y1[:20]))
                        # y2 = model.extract_embed(
                        #     xx[:,:,25:1037],
                        #     chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 1], y2[:20]))
                        # y3 = model.extract_embed(
                        #      xx[:,:,250:1262],
                        #      chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 10], y3[:20]))

                        # y3 = model.extract_embed(
                        #      xx[:,:,250:1262],
                        #      chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 10], y3[:20]))

                        # y3 = model.extract_embed(
                        #      xx[:,:,2500:3512],
                        #      chunk_length=chunk_length,
                        #     embed_layer=embed_layer, detach_chunks=True).detach().cpu().numpy()[0]
                        # logging.info('{} {}'.format(y[:20, 100], y3[:20]))

                t5 = time.time()
                y = y.T
                writer.write(key, [y])

                if write_timestamps_spec is not None:
                    num_wins = y.shape[0]
                    timestamps = model.compute_slidwin_timestamps(
                        num_wins,
                        win_length,
                        win_shift,
                        snip_edges,
                        feat_frame_length,
                        feat_frame_length,
                        feat_snip_edges,
                    ).numpy()
                    logging.info("{}".format(timestamps))
                    time_writer.write(key, [timestamps])
                t6 = time.time()
                logging.info(
                    (
                        "utt %s total-time=%.3f read-time=%.3f mvn-time=%.3f "
                        "vad-time=%.3f embed-time=%.3f write-time=%.3f "
                        "rt-factor=%.2f"
                    )
                    % (
                        key[0],
                        t6 - t1,
                        t2 - t1,
                        t3 - t2,
                        t4 - t3,
                        t5 - t4,
                        t6 - t5,
                        x.shape[0] * 1e-2 / (t6 - t1),
                    )
                )

    if write_timestamps_spec is not None:
        time_writer.close()

    if slidwin_params_path is not None:
        params = {
            "padding": model.compute_slidwin_left_padding(
                win_length,
                win_shift,
                snip_edges,
                feat_frame_length,
                feat_frame_length,
                feat_snip_edges,
            ),
            "win_length": win_length,
            "win_shift": win_shift,
        }
        with open(slidwin_params_path, "w") as f:
            yaml.dump(params, f)


if __name__ == "__main__":

    parser = ArgumentParser(description="Extract x-vectors over a sliding window")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    DRF.add_class_args(parser)
    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument(
        "--write-timestamps", dest="write_timestamps_spec", default=None
    )
    parser.add_argument("--slidwin-params-path", default=None)

    # parser.add_argument('--scp-sep', dest='scp_sep', default=' ',
    #                     help=('scp file field separator'))
    # parser.add_argument('--path-prefix', dest='path_prefix', default=None,
    #                     help=('scp file_path prefix'))
    parser.add_argument(
        "--vad-path-prefix",
        dest="vad_path_prefix",
        default=None,
        help=("scp file_path prefix for vad"),
    )

    MVN.add_class_args(parser, prefix="mvn")

    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--win-length",
        type=float,
        default=1.5,
        help=("window length for x-vector extraction in seconds"),
    )
    parser.add_argument(
        "--win-shift",
        type=float,
        default=0.25,
        help=("window shift for x-vector extraction in seconds"),
    )
    parser.add_argument(
        "--snip-edges",
        default=False,
        action="store_true",
        help=(
            "If true, end effects will be handled by outputting "
            "only windows that completely fit in the file, "
            "and the number of windows depends on the window-length. "
            "If false, the number of windows depends only on "
            "the window-shift, and we reflect the data at the ends."
        ),
    )

    parser.add_argument(
        "--feat-frame-length",
        type=float,
        default=25,
        help=("frame-length used to compute the acoustic features in msecs"),
    )
    parser.add_argument(
        "--feat-frame-shift",
        type=float,
        default=10,
        help=("frame-shift used to compute the acoustic features in msecs"),
    )
    parser.add_argument(
        "--feat-snip-edges",
        default=False,
        action="store_true",
        help=(
            "If true, end effects will be handled by outputting only windows "
            "that completely fit in the file, and the number of windows "
            "depends on the feat-frame-length. "
            "If false, the number of feature frames depends only on the "
            "feat-frame-shift, and we reflect the waveform at the ends."
        ),
    )

    parser.add_argument(
        "--chunk-length",
        type=int,
        default=0,
        help=(
            "number of frames used in each forward pass of the x-vector encoder,"
            "if 0 the full utterance is used"
        ),
    )

    parser.add_argument(
        "--embed-layer",
        type=int,
        default=None,
        help=(
            "classifier layer to get the embedding from,"
            "if None the layer set in training phase is used"
        ),
    )

    parser.add_argument("--output", dest="output_spec", required=True)
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    # parser.add_argument('--part-idx', dest='part_idx', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    # parser.add_argument('--num-parts', dest='num_parts', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_xvectors(**namespace_to_dict(args))

#!/usr/bin/env python
"""
 Copyright 2020 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import time
import logging
from pathlib import Path
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from hyperion.hyp_defs import config_logger, float_cpu, set_float_cpu
from hyperion.utils import Utt2Info
from hyperion.io import DataWriterFactory as DWF
from hyperion.io import SequentialDataReaderFactory as DRF
from hyperion.io import VADReaderFactory as VRF
from hyperion.np.feats import MeanVarianceNorm as MVN

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
    logging.info("vae-model={}".format(model))
    model.to(device)
    model.eval()
    return model


def write_img(output_dir, key, x, x_mean, x_sample, num_frames):

    vmax = np.max(x)
    vmin = np.min(x)
    if x.shape[1] > num_frames:
        x = x[:, :num_frames]
        x_mean = x_mean[:, :num_frames]
        x_sample = x_sample[:, :num_frames]
    elif x.shape[1] < num_frames:
        x_extra = vmin * np.ones(
            (x.shape[0], num_frames - x.shape[1]), dtype=float_cpu()
        )
        x = np.concatenate((x, x_extra), axis=1)
        x_mean = np.concatenate((x_mean, x_extra), axis=1)
        x_sample = np.concatenate((x_sample, x_extra), axis=1)

    cmap = plt.get_cmap("jet")
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1
    )
    plt.figure(figsize=(12, 8), dpi=300)
    plt.subplot(3, 1, 1)
    plt.imshow(x, aspect=2, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.subplot(3, 1, 2)
    plt.imshow(x_mean, aspect=2, cmap=cmap, vmax=vmax, vmin=vmin)
    plt.subplot(3, 1, 3)
    plt.imshow(x_sample, aspect=2, cmap=cmap, vmax=vmax, vmin=vmin)

    file_path = Path(output_dir, key + ".pdf")
    plt.savefig(file_path)
    plt.close()


def eval_vae(
    input_spec,
    vad_spec,
    write_num_frames_spec,
    vad_path_prefix,
    model_path,
    score_path,
    write_x_mean_spec,
    write_x_sample_spec,
    write_z_sample_spec,
    write_img_path,
    img_frames,
    use_gpu,
    **kwargs
):

    logging.info("initializing")
    rng = np.random.RandomState(seed=1123581321 + kwargs["part_idx"])
    device = init_device(use_gpu)
    mvn = init_mvn(device, **kwargs)
    model = load_model(model_path, device)

    if write_num_frames_spec is not None:
        keys = []
        info = []

    # num_gpus = 1 if use_gpu else 0
    # logging.info('initializing devices num_gpus={}'.format(num_gpus))
    # device = open_device(num_gpus=num_gpus)
    # logging.info('loading model {}'.format(model_path))
    # model = TML.load(model_path)
    # model.to(device)
    # model.eval()

    x_mean_writer = None
    x_sample_writer = None
    z_sample_writer = None
    fargs = {"return_x_mean": True}  # args for forward function
    if write_x_mean_spec is not None:
        logging.info("opening write x-mean stream: %s" % (write_x_mean_spec))
        x_mean_writer = DWF.create(write_x_mean_spec)

    if write_x_sample_spec is not None:
        logging.info("opening write x-sample stream: %s" % (write_x_sample_spec))
        x_sample_writer = DWF.create(write_x_sample_spec)
        fargs["return_x_sample"] = True

    if write_z_sample_spec is not None:
        logging.info("opening write z-sample stream: %s" % (write_z_sample_spec))
        z_sample_writer = DWF.create(write_z_sample_spec)
        fargs["return_z_sample"] = True

    if write_img_path is not None:
        logging.info("making img dir: %s" % (write_img_path))
        fargs["return_x_mean"] = True
        fargs["return_x_sample"] = True
        Path(write_img_path).mkdir(parents=True, exist_ok=True)

    metrics = ["loss", "elbo", "log_px", "kldiv_z", "vq_loss", "log_perplexity"]
    extra_metrics = {"mse": nn.MSELoss(), "L1": nn.L1Loss()}
    scores_df = []

    dr_args = DRF.filter_args(**kwargs)
    logging.info("opening input stream: %s" % (input_spec))
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
            scores = {"key": key[0]}
            if x.shape[0] == 0:
                x_mean, x_sample = np.zeros((1, x.shape[1]), dtype=float_cpu())
                z_sample = np.zeros((1, model.z_dim), dtype=float_cpu())
            else:
                xx = torch.tensor(x.T[None, :], dtype=torch.get_default_dtype())
                with torch.no_grad():
                    xx = xx.to(device)
                    output = model(xx, **fargs)

                    x_mean = output["x_mean"]
                    for metric in metrics:
                        if metric in output:
                            scores[metric] = output[metric].mean().item()

                    for metric in extra_metrics.keys():
                        scores[metric] = extra_metrics[metric](x_mean, xx).item()

                    # elbo = elbo.mean().item()
                    # log_px = log_px.mean().item()
                    # kldiv_z = kldiv_z.mean().item()
                    # mse = nn.functional.mse_loss(px.mean, xx).item()
                    # l1 = nn.functional.l1_loss(px.mean, xx).item()

                    logging.info("utt {} scores={}".format(key[0], scores))

                    # logging.info('utt %s elbo=%.2f E[logP(x|z)]=%.2f KL(q(z)||p(z))=%.2f mse=%.2f l1=%.2f' % (
                    #    key[0], elbo, log_px, kldiv_z, mse, l1))

                    x_mean = x_mean.cpu().numpy()[0]
                    if "x_sample" in output:
                        x_sample = output["x_sample"].cpu().numpy()[0]
                    if "z_sample" in output:
                        z_sample = output["z_mean"].cpu().numpy()[0]

                if write_img_path:
                    write_img(write_img_path, key[0], x.T, x_mean, x_sample, img_frames)

            t5 = time.time()
            scores_df.append(pd.DataFrame(scores, index=[0]))
            if x_mean_writer is not None:
                x_mean_writer.write(key, [x_mean.T])
            if x_sample_writer is not None:
                x_sample_writer.write(key, [x_sample.T])
            if z_sample_writer is not None:
                z_sample_writer.write(key, [z_sample.T])

            if write_num_frames_spec is not None:
                keys.append(key[0])
                info.append(str(x.shape[0]))
            t6 = time.time()
            logging.info(
                (
                    "utt %s total-time=%.3f read-time=%.3f mvn-time=%.3f "
                    "vad-time=%.3f vae-time=%.3f write-time=%.3f "
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

    scores_df = pd.concat(scores_df, ignore_index=True)
    scores_df.to_csv(score_path, index=False, na_rep="n/a")

    if write_num_frames_spec is not None:
        logging.info("writing num-frames to %s" % (write_num_frames_spec))
        u2nf = Utt2Info.create(keys, info)
        u2nf.save(write_num_frames_spec)


if __name__ == "__main__":

    parser = ArgumentParser(description="Extract x-vectors with pytorch model")

    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--input", dest="input_spec", required=True)
    DRF.add_class_args(parser)
    parser.add_argument("--vad", dest="vad_spec", default=None)
    parser.add_argument(
        "--write-num-frames", dest="write_num_frames_spec", default=None
    )
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
    # parser.add_argument('--chunk-length', type=int, default=0,
    #                     help=('number of frames used in each forward pass of the x-vector encoder,'
    #                           'if 0 the full utterance is used'))

    parser.add_argument(
        "--write-x-mean",
        dest="write_x_mean_spec",
        default=None,
        help="write-specifier for the mean of P(x|z)",
    )
    parser.add_argument(
        "--write-x-sample",
        dest="write_x_sample_spec",
        default=None,
        help="write-specifier for samples drawn from x ~ P(x|z)",
    )
    parser.add_argument(
        "--write-z-sample",
        dest="write_z_sample_spec",
        default=None,
        help="write-specifier for samples drawn from z ~ Q(z|x)",
    )
    parser.add_argument(
        "--write-img-path",
        default=None,
        help="output directory to save spectrogram images in pdf format",
    )
    parser.add_argument(
        "--img-frames",
        default=400,
        type=int,
        help="number of frames to plot in the images",
    )
    parser.add_argument(
        "--scores",
        dest="score_path",
        required=True,
        help="output file to write ELBO and other metrics",
    )
    parser.add_argument(
        "--use-gpu", default=False, action="store_true", help="extract xvectors in gpu"
    )
    # parser.add_argument('--part-idx', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    # parser.add_argument('--num-parts', type=int, default=1,
    #                     help=('splits the list of files in num-parts and process part_idx'))
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    eval_vae(**namespace_to_dict(args))

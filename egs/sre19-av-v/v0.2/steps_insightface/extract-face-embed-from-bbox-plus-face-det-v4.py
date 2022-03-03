#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import sys
import os
import argparse
import time
import logging

# import copy

import numpy as np
import pandas as pd

# import av
# import cv2
import h5py

import torch

from hyperion.hyp_defs import config_logger

from retinaface.detector import RetinafaceDetector
from models import resnet101

from hyperion.utils import SCPList

from face_utils import (
    read_video_windows,
    save_facedet_image,
    save_facecrop_images,
    save_facealign_images,
    save_bbox,
    detect_faces_in_frame,
    extract_embed_in_frame_v4,
    select_face,
)


def extract_face_embed(
    input_path,
    bbox_path,
    output_path,
    facedet_model_file,
    faceembed_model_file,
    fps,
    min_faces,
    use_gpu,
    save_facedet_img,
    save_facecrop_img,
    save_facealign_img,
    time_in_secs,
    det_window,
    thr_overlap,
    thr_d,
    part_idx,
    num_parts,
):

    scp = SCPList.load(input_path)
    if num_parts > 0:
        scp = scp.split(part_idx, num_parts)

    output_dir = os.path.dirname(output_path)

    device_type = "cuda" if use_gpu else "cpu"
    detector = RetinafaceDetector("mnet", facedet_model_file, device_type)

    f_bb = open(output_path + ".bbox", "w")
    f_scp = open(output_path + ".scp", "w")
    h5_file = output_path + ".h5"
    f_h5 = h5py.File(h5_file, "w")

    bbox_table = pd.read_csv(
        bbox_path,
        sep=" ",
        header=None,
        names=["key", "idx", "x1", "y1", "x2", "y2"],
        usecols=[0, 1, 2, 3, 4, 5],
    )

    class HParams:
        def __init__(self):
            self.pretrained = False
            self.use_se = True

    config = HParams()
    embed_extractor = resnet101(config)
    embed_extractor.load_state_dict(
        torch.load(faceembed_model_file, map_location="cpu")
    )
    device = torch.device(device_type)
    embed_extractor = embed_extractor.to(device)
    embed_extractor.eval()

    facedet_dir = None
    facecrop_dir = None
    facealign_dir = None
    dummy_face = torch.zeros((1, 3, 112, 112), dtype=torch.float, device=device)
    with torch.no_grad():
        x = embed_extractor(dummy_face)
    x_dim = x.shape[-1]
    logging.info("embed dim=%d", x_dim)

    for key, file_path, _, _ in scp:
        if save_facedet_img:
            facedet_dir = "%s/img_facedet/%s" % (output_dir, key)
            if not os.path.exists(facedet_dir):
                os.makedirs(facedet_dir)

        if save_facecrop_img:
            facecrop_dir = "%s/img_facecrop/%s" % (output_dir, key)
            if not os.path.exists(facecrop_dir):
                os.makedirs(facecrop_dir)

        if save_facealign_img:
            facealign_dir = "%s/img_facealign/%s" % (output_dir, key)
            if not os.path.exists(facealign_dir):
                os.makedirs(facealign_dir)

        # get bboxes for this file:
        file_bbox = bbox_table.loc[bbox_table["key"] == key]
        assert np.all(np.sort(file_bbox["idx"]) == file_bbox["idx"]), str(
            file_bbox["idx"]
        )
        time_idx = file_bbox["idx"]

        logging.info("loading video %s from path %s", key, file_path)
        t1 = time.time()
        frames, frame_idx, window_idx = read_video_windows(
            file_path, time_idx, det_window, time_in_secs
        )
        dt = time.time() - t1
        logging.info("loading time %.2f", dt)
        threshold = 0.9
        while threshold > 0.01:
            x = []
            overlap_scores = []
            d_scores = []
            for frame, idx, w_idx in zip(frames, frame_idx, window_idx):
                logging.info(
                    "processing file %s frame %d of shape=%s",
                    key,
                    idx,
                    str(frame.shape),
                )
                faces, landmarks = detect_faces_in_frame(
                    detector, frame, thresh=threshold
                )
                logging.info(
                    "file %s frame %d dectected %d faces", key, idx, faces.shape[0]
                )

                if save_facedet_img:
                    save_facedet_image(key, idx, frame, faces, landmarks, facedet_dir)

                if faces.shape[0] == 0:
                    continue

                frame_bbox = file_bbox.iloc[w_idx]
                ref_face = np.asarray(frame_bbox[["x1", "y1", "x2", "y2"]])
                print("faces", ref_face, faces, flush=True)
                best_face, faces, overlap_score, d_score = select_face(faces, ref_face)

                # logging.info('file %s frame %d selected face with overlap-score=%.2f d-score=%.2f', key, idx, overlap_score, d_score)
                if overlap_score < thr_overlap and d_score > thr_d:
                    continue

                logging.info(
                    "file %s frame %d selected face with overlap-score=%.2f d-score=%.2f",
                    key,
                    idx,
                    overlap_score,
                    d_score,
                )
                faces = np.expand_dims(faces, axis=0)
                landmarks = np.expand_dims(landmarks[best_face], axis=0)
                overlap_score = np.expand_dims(overlap_score, axis=0)
                d_score = np.expand_dims(d_score, axis=0)

                x_f, q_f = extract_embed_in_frame_v4(
                    embed_extractor,
                    frame,
                    landmarks,
                    thresh=threshold,
                    x_dim=x_dim,
                    device=device,
                )

                logging.info(
                    "file %s frame %d extracted %d face embeds",
                    key,
                    idx,
                    faces.shape[0],
                )
                print(x_f.shape, overlap_score.shape, d_score.shape, flush=True)
                x.append(x_f)
                overlap_scores.append(overlap_score)
                d_scores.append(d_score)

                if faces.shape[0] == 0:
                    continue

                if save_facecrop_img:
                    save_facecrop_images(key, idx, frame, faces, facecrop_dir)
                if save_facealign_img:
                    save_facealign_images(key, idx, frame, landmarks, facealign_dir)

                save_bbox(key, idx, faces, f_bb)

            if len(x) > 0:
                x = np.concatenate(tuple(x), axis=0)
            else:
                x = np.zeros((0, x_dim))

            if min_faces == 0 or x.shape[0] >= min_faces:
                overlap_scores = np.concatenate(tuple(overlap_scores))
                d_scores = np.concatenate(tuple(d_scores))
                break

            threshold -= 0.1
            logging.info(
                "did not detect faces in file %s, reducing facedet threshold=%.1f",
                key,
                threshold,
            )

        # select faces given the overlap scores
        if x.shape[0] > 0:
            min_overlap_score = np.max(overlap_scores) / 2
            max_d_score = 0
            min_d_score = 0
            if min_overlap_score > 0:
                sel_idx = overlap_scores > min_overlap_score
            else:
                max_d_score = np.max(d_scores)
                min_d_score = 0.8 * max_d_score
                sel_idx = d_scores > min_d_score

            logging.info(
                "file %s select %d faces out of %d max-overlap-score=%.3f min-overlap-score=%.3f max-d-score=%.2f min-d-score=%.2f",
                key,
                np.sum(sel_idx),
                x.shape[0],
                2 * min_overlap_score,
                min_overlap_score,
                max_d_score,
                min_d_score,
            )
            x = x[sel_idx]
        logging.info("file %s saving %d face embeds", key, x.shape[0])
        f_scp.write("%s %s\n" % (key, h5_file))
        f_h5.create_dataset(key, data=x.astype("float32"))

    f_bb.close()
    f_scp.close()
    f_h5.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Face embedding extractory with Pytorch RetinaFace and ArcFace",
    )

    parser.add_argument("--input-path", required=True)
    parser.add_argument("--bbox-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--facedet-model-file", required=True)
    parser.add_argument("--faceembed-model-file", required=True)
    parser.add_argument("--use-gpu", default=False, action="store_true")
    parser.add_argument(
        "--save-facedet-img",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-facecrop-img",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save-facealign-img",
        default=False,
        action="store_true",
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--time-in-secs", default=False, action="store_true")
    parser.add_argument("--det-window", type=int, default=21)
    parser.add_argument("--thr-overlap", type=float, default=0.001)
    parser.add_argument("--thr-d", type=float, default=0)
    parser.add_argument("--min-faces", type=int, default=1)
    parser.add_argument("--part-idx", type=int, default=1)
    parser.add_argument("--num-parts", type=int, default=1)

    parser.add_argument(
        "-v",
        "--verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
        help="Verbose level",
    )
    args = parser.parse_args()
    config_logger(args.verbose)
    del args.verbose
    logging.debug(args)

    extract_face_embed(**vars(args))

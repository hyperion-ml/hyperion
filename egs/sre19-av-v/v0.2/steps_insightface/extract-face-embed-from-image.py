#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import os
import argparse
import time
import logging

import numpy as np

# import av
# import cv2
import h5py

import torch

from hyperion.hyp_defs import config_logger

from retinaface.detector import RetinafaceDetector
from models import resnet101

from hyperion.utils import SCPList


from face_utils import (
    read_img,
    save_facedet_image,
    save_facecrop_images,
    save_facealign_images,
    save_bbox,
    detect_faces_in_frame,
    extract_embed_in_frame_v4,
)


def extract_face_embed(
    input_path,
    output_path,
    facedet_model_file,
    faceembed_model_file,
    min_faces,
    use_gpu,
    save_facedet_img,
    save_facecrop_img,
    save_facealign_img,
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

    dummy_face = torch.zeros((1, 3, 112, 112), dtype=torch.float, device=device)
    with torch.no_grad():
        x = embed_extractor(dummy_face)
    x_dim = x.shape[-1]
    logging.info("embed dim=%d", x_dim)

    facedet_dir = None
    facecrop_dir = None
    facealign_dir = None
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

        logging.info("loading video %s from path %s", key, file_path)
        t1 = time.time()
        frame = read_img(file_path)
        dt = time.time() - t1
        logging.info("loading time %.2f", dt)
        threshold = 0.9
        while threshold > 0.01:
            logging.info(
                "processing file %s of shape=%s",
                key,
                str(frame.shape),
            )
            faces, landmarks = detect_faces_in_frame(detector, frame, thresh=threshold)
            logging.info("file %s detected %d faces", key, faces.shape[0])

            if save_facedet_img:
                save_facedet_image(key, 0, frame, faces, landmarks, facedet_dir)

            if faces.shape[0] == 0:
                threshold -= 0.1
                logging.info(
                    "did not detect faces in file %s, reducing facedet threshold=%.1f",
                    key,
                    threshold,
                )
                continue

            x, _ = extract_embed_in_frame_v4(
                embed_extractor,
                frame,
                landmarks,
                thresh=threshold,
                x_dim=x_dim,
                device=device,
            )
            logging.info(
                "file %s extracted %d face embeds",
                key,
                faces.shape[0],
            )

            if save_facecrop_img:
                save_facecrop_images(key, 0, frame, faces, facecrop_dir)
            if save_facealign_img:
                save_facealign_images(key, 0, frame, landmarks, facealign_dir)

            save_bbox(key, 0, faces, f_bb)

            if min_faces == 0 or x.shape[0] >= min_faces:
                break

        logging.info("file %s saving %d face embeds", key, x.shape[0])
        f_scp.write("%s %s\n" % (key, h5_file))
        f_h5.create_dataset(key, data=x.astype("float32"))

    f_bb.close()
    f_scp.close()
    f_h5.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Face detection with Insightface Retina model",
    )

    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--facedet-model-file", required=True)
    parser.add_argument("--faceembed-model-file", required=True)
    parser.add_argument("--use-gpu", default=False, action="store_true")
    parser.add_argument("--save-facedet-img", default=False, action="store_true")
    parser.add_argument("--save-facecrop-img", default=False, action="store_true")
    parser.add_argument("--save-facealign-img", default=False, action="store_true")
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

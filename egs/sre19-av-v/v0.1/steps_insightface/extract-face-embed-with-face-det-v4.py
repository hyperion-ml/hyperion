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

from retinaface import RetinaFace
from face_model import FaceModel

from scp_list import SCPList

from face_utils import (
    Namespace,
    read_video,
    save_facedet_image,
    save_facecrop_images,
    save_bbox,
    detect_faces_in_frame,
    extract_embed_in_frame_v4,
)


def extract_face_embed(
    input_path,
    output_path,
    facedet_model_file,
    faceembed_model_file,
    fps,
    min_faces,
    use_gpu,
    save_facedet_img,
    save_facecrop_img,
    part_idx,
    num_parts,
):

    scp = SCPList.load(input_path)
    if num_parts > 0:
        scp = scp.split(part_idx, num_parts)

    output_dir = os.path.dirname(output_path)

    gpu_id = 0 if use_gpu else -1
    detector = RetinaFace(facedet_model_file, 0, gpu_id, "net3")
    f_bb = open(output_path + ".bbox", "w")
    f_scp = open(output_path + ".scp", "w")
    h5_file = output_path + ".h5"
    f_h5 = h5py.File(h5_file, "w")

    faceembed_model_file = "%s,0" % (faceembed_model_file)
    args_face_model = Namespace(
        threshold=1.24,
        model=faceembed_model_file,
        ga_model="",
        image_size="112,112",
        gpu=gpu_id,
        det=1,
        flip=0,
    )
    embed_extractor = FaceModel(args_face_model)

    dummy_face = np.random.rand(3, 112, 112).astype(dtype=np.uint)
    x_i = embed_extractor.get_feature(dummy_face)
    x_dim = x_i.shape[0]
    logging.info("embed dim=%d", x_dim)

    facedet_dir = None
    facecrop_dir = None
    for key, file_path, _, _ in scp:
        if save_facedet_img:
            facedet_dir = "%s/img_facedet/%s" % (output_dir, key)
            if not os.path.exists(facedet_dir):
                os.makedirs(facedet_dir)

        if save_facecrop_img:
            facecrop_dir = "%s/img_facecrop/%s" % (output_dir, key)
            if not os.path.exists(facecrop_dir):
                os.makedirs(facecrop_dir)

        logging.info("loading video %s from path %s", key, file_path)
        t1 = time.time()
        frames, frame_idx = read_video(file_path, fps)
        dt = time.time() - t1
        logging.info("loading time %.2f", dt)
        threshold = 0.9
        use_retina_landmarks = False
        while threshold > 0.01:
            x = []
            retina_detect_faces = False
            for frame, idx in zip(frames, frame_idx):
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

                retina_detect_faces = True

                x_f, valid = extract_embed_in_frame_v4(
                    embed_extractor,
                    frame,
                    faces,
                    landmarks,
                    thresh=threshold,
                    use_retina=use_retina_landmarks,
                    x_dim=x_dim,
                )
                x_f = x_f[valid]
                faces = faces[valid]
                logging.info(
                    "file %s frame %d extracted %d face embeds",
                    key,
                    idx,
                    faces.shape[0],
                )
                x.append(x_f)

                if save_facecrop_img:
                    save_facecrop_images(key, idx, frame, faces, facecrop_dir)

                save_bbox(key, idx, faces, f_bb)

            if len(x) > 0:
                x = np.concatenate(tuple(x), axis=0)
            else:
                x = np.zeros((0, x_dim))

            if min_faces == 0 or x.shape[0] >= min_faces:
                break

            if retina_detect_faces and not use_retina_landmarks:
                use_retina_landmarks = True
                logging.info(
                    (
                        "retina face detected faces in file %s but mtcnn did not, "
                        "using retina landmarks then"
                    ),
                    key,
                )
            else:
                threshold -= 0.1
                logging.info(
                    "did not detect faces in file %s, reducing facedet threshold=%.1f",
                    key,
                    threshold,
                )

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
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--min-faces", type=int, default=1)
    parser.add_argument("--part-idx", type=int, default=1)
    parser.add_argument("--num-parts", type=int, default=1)

    # parser.add_argument('-v', '--verbose', default=1, choices=[0, 1, 2, 3], type=int,
    #                    help='Verbose level')
    args = parser.parse_args()
    # config_logger(args.verbose)
    # del args.verbose
    # logging.debug(args)
    logging.basicConfig(
        level=2, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )

    extract_face_embed(**vars(args))

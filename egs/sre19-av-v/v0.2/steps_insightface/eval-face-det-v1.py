#!/usr/bin/env python
"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import os
import argparse
import time
import logging
import copy

import numpy as np

import av
import cv2

from retinaface.detector import RetinafaceDetector
from utils import align_face
from hyperion.utils import SCPList


def read_video_frames(file_path, fps):

    f = av.open(str(file_path))
    f.streams.video[0].thread_type = "AUTO"  # Go faster!
    video_fps = (
        float(f.streams.video[0].average_rate.numerator)
        / f.streams.video[0].average_rate.denominator
    )
    delta = video_fps / fps
    frames = []
    frame_idx = []
    next_frame = 0
    for count, frame in enumerate(f.decode(video=0)):
        if count == int(np.round(next_frame)):
            frame_array = frame.to_ndarray(format="rgb24")
            # IMPORTANT!!!!!!
            # OpenCV uses BGR channel order, we need to flip the last dimension!!!
            frame_array = copy.deepcopy(frame_array[:, :, ::-1])
            frames.append(frame_array)
            frame_idx.append(count)
            next_frame += delta

    return frames, frame_idx


def resize_img(img):
    """Resizes the image so the long axis is lower or eq to 800"""
    max_size = 800
    ratio = 1
    h, w = img.shape[:2]

    if h > max_size or w > max_size:
        if h > w:
            ratio = max_size / h
        else:
            ratio = max_size / w

        img = cv2.resize(img, (int(round(w * ratio)), int(round(h * ratio))))
    return img, ratio


def eval_face_det(
    input_path,
    output_path,
    model_file,
    fps,
    use_gpu,
    save_face_img,
    part_idx,
    num_parts,
):

    scp = SCPList.load(input_path)
    if num_parts > 0:
        scp = scp.split(part_idx, num_parts)

    output_dir = os.path.dirname(output_path)

    thresh = 0.9

    device_type = "cuda" if use_gpu else "cpu"
    detector = RetinafaceDetector("mnet", model_file, device_type)
    f_bb = open(output_path, "w")

    for key, file_path, _, _ in scp:
        if save_face_img:
            output_dir_i = "%s/img/%s" % (output_dir, key)
            if not os.path.exists(output_dir_i):
                os.makedirs(output_dir_i)

        logging.info("loading video %s from path %s" % (key, file_path))
        t1 = time.time()
        frames, frame_idx = read_video_frames(file_path, fps)
        dt = time.time() - t1
        logging.info("loading time %.2f" % dt)
        for frame, idx in zip(frames, frame_idx):
            logging.info(
                "processing file %s frame %d of shape=%s" % (key, idx, str(frame.shape))
            )
            frame, ratio = resize_img(frame)
            faces, landmarks = detector.detect_faces(frame, thresh)
            faces = faces / ratio
            landmarks = landmarks / ratio

            logging.info(
                "file %s frame %d dectected %d faces" % (key, idx, faces.shape[0])
            )
            print(faces, landmarks, flush=True)
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int32)
                logging.info("file %s frame %d bb=%s" % (key, idx, str(box)))
                f_bb.write(
                    "%s %d %d %d %d %d\n" % (key, idx, box[0], box[1], box[2], box[3])
                )

            if save_face_img:
                for i in range(faces.shape[0]):
                    # print('score', faces[i][4])
                    box = faces[i].astype(np.int32)
                    # color = (255,0,0)
                    color = (0, 0, 255)
                    frame0 = copy.deepcopy(frame)
                    logging.info("file %s frame %d bb=%s" % (key, idx, str(box)))
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    # logging.info(str(np.max(np.abs(frame0 - frame))))
                    if landmarks is not None:
                        landmark5 = landmarks[i].astype(np.int32)
                        # print(landmark.shape)
                        for l in range(landmark5.shape[0] // 2):
                            color = (0, 0, 255)
                            if l == 0 or l == 1:
                                color = (0, 255, 0)
                            cv2.circle(
                                frame, (landmark5[l], landmark5[l + 5]), 1, color, 2
                            )
                        # align and save
                        frame_align = align_face(frame0, [landmark5])
                        img_filename = "%s/align_%04d_%02d.jpeg" % (
                            output_dir_i,
                            idx,
                            i,
                        )
                        logging.info(
                            "file %s align frame %d fade %d saved to %s",
                            key,
                            idx,
                            i,
                            img_filename,
                        )
                        cv2.imwrite(img_filename, frame_align)

                img_filename = "%s/%04d.jpeg" % (output_dir_i, idx)
                logging.info("file %s frame %d saved to %s" % (key, idx, img_filename))
                cv2.imwrite(img_filename, frame)

    f_bb.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Face detection with Insightface Retina model",
    )

    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--use-gpu", default=False, action="store_true")
    parser.add_argument("--save-face-img", default=False, action="store_true")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--part-idx", type=int, default=1)
    parser.add_argument("--num-parts", type=int, default=1)

    # parser.add_argument('-v', '--verbose', dest='verbose', default=1, choices=[0, 1, 2, 3], type=int,
    #                    help='Verbose level')
    args = parser.parse_args()
    # config_logger(args.verbose)
    # del args.verbose
    # logging.debug(args)
    logging.basicConfig(
        level=2, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )

    eval_face_det(**vars(args))

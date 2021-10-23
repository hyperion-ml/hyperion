"""
 Copyright 2019 Jesus Villalba (Johns Hopkins University)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""

import sys
import os
import argparse
import time
import logging
import copy

import numpy as np

import av
import cv2
import h5py

import torch

from hyperion.utils import SCPList

from torchvision import transforms
from retinaface.detector import RetinafaceDetector
from utils import align_face

rotation_opts = {
    "90": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def read_img(file_path):
    return cv2.imread(file_path)


def read_video(file_path, fps):

    f = av.open(str(file_path))
    f.streams.video[0].thread_type = "AUTO"  # Go faster!
    video_fps = (
        float(f.streams.video[0].average_rate.numerator)
        / f.streams.video[0].average_rate.denominator
    )
    delta = video_fps / fps
    meta = f.streams.video[0].metadata
    rotate = -1
    if "rotate" in meta:
        rotate = rotation_opts[meta["rotate"]]

    frames = []
    frame_idx = []
    next_frame = 0
    for count, frame in enumerate(f.decode(video=0)):
        if count == int(np.round(next_frame)):
            frame_array = frame.to_ndarray(format="rgb24")
            # IMPORTANT!!!!!!
            # OpenCV uses BGR channel order, we need to flip the last dimension!!!
            frame_array = copy.deepcopy(frame_array[:, :, ::-1])
            if rotate != -1:
                frame_array = cv2.rotate(frame_array, rotate)

            frames.append(frame_array)
            frame_idx.append(count)
            next_frame += delta

    return frames, frame_idx


def read_video_frames(file_path, frame_idx=None, time_in_secs=False):

    f = av.open(str(file_path))
    f.streams.video[0].thread_type = "AUTO"  # Go faster!
    video_fps = (
        float(f.streams.video[0].average_rate.numerator)
        / f.streams.video[0].average_rate.denominator
    )
    if time_in_secs:
        # transform seconds into frames indexes
        frame_idx = [int(s * video_fps) for s in frame_idx]

    meta = f.streams.video[0].metadata
    rotate = -1
    if "rotate" in meta:
        rotate = rotation_opts[meta["rotate"]]

    frames = []
    k = 0
    next_frame = frame_idx[k]
    for count, frame in enumerate(f.decode(video=0)):
        if count == int(np.round(next_frame)):
            frame_array = frame.to_ndarray(format="rgb24")
            # IMPORTANT!!!!!!
            # OpenCV uses BGR channel order, we need to flip the last dimension!!!
            frame_array = copy.deepcopy(frame_array[:, :, ::-1])
            if rotate != -1:
                frame_array = cv2.rotate(frame_array, rotate)

            frames.append(frame_array)
            k += 1
            if k == len(frame_idx):
                break
            next_frame = frame_idx[k]

    return frames, frame_idx


def read_video_windows(file_path, frame_idx=None, det_window=30, time_in_secs=False):

    context = int((det_window - 1) / 2)
    f = av.open(str(file_path))
    f.streams.video[0].thread_type = "AUTO"  # Go faster!
    video_fps = (
        float(f.streams.video[0].average_rate.numerator)
        / f.streams.video[0].average_rate.denominator
    )
    if time_in_secs:
        # transform seconds into frames indexes
        frame_idx = [int(s * video_fps) for s in frame_idx]

    meta = f.streams.video[0].metadata
    rotate = -1
    if "rotate" in meta:
        rotate = rotation_opts[meta["rotate"]]

    all_frame_idx = []
    window_idx = []
    last_frame_p1 = 0
    for i, t in enumerate(frame_idx):
        first_frame = max(last_frame_p1, t - context)
        last_frame_p1 = t + context + 1
        n_frames = last_frame_p1 - first_frame
        frame_idx_i = np.arange(first_frame, last_frame_p1, dtype=np.int)
        window_idx_i = i * np.ones((n_frames,), dtype=np.int)
        all_frame_idx.append(frame_idx_i)
        window_idx.append(window_idx_i)

    frame_idx = np.concatenate(tuple(all_frame_idx))
    window_idx = np.concatenate(tuple(window_idx))
    frames = []
    k = 0
    next_frame = frame_idx[k]
    for count, frame in enumerate(f.decode(video=0)):
        if count == int(np.round(next_frame)):
            frame_array = frame.to_ndarray(format="rgb24")
            # IMPORTANT!!!!!!
            # OpenCV uses BGR channel order, we need to flip the last dimension!!!
            frame_array = copy.deepcopy(frame_array[:, :, ::-1])
            if rotate != -1:
                frame_array = cv2.rotate(frame_array, rotate)

            frames.append(frame_array)
            k += 1
            if k == len(frame_idx):
                break
            next_frame = frame_idx[k]

    return frames, frame_idx, window_idx


def save_facedet_image(key, idx, frame, faces, landmarks, facedet_dir):

    frame = copy.deepcopy(frame)
    for i in range(faces.shape[0]):
        # print('score', faces[i][4])
        bbox = faces[i].astype(np.int)
        # color = (255,0,0)
        color = (0, 0, 255)
        logging.info("file %s frame %d bb=%s" % (key, idx, str(bbox)))
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        if landmarks is not None:
            landmark5 = landmarks[i].astype(np.int)
            for l in range(landmark5.shape[0] // 2):
                color = (0, 0, 255)
                if l == 0 or l == 1:
                    color = (0, 255, 0)
                cv2.circle(frame, (landmark5[l], landmark5[l + 5]), 1, color, 2)

    img_filename = "%s/%06d.jpeg" % (facedet_dir, idx)
    logging.info("file %s frame %d saved to %s" % (key, idx, img_filename))
    cv2.imwrite(img_filename, frame)


def save_facecrop_images(key, idx, frame, landmarks, facecrop_dir):

    frame = copy.deepcopy(frame)
    for i in range(faces.shape[0]):
        # print('score', faces[i][4])
        bbox = faces[i].astype(np.int)
        frame_i = frame[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
        img_filename = "%s/%06d-%02d.jpeg" % (facecrop_dir, idx, i)
        logging.info(
            "file %s frame %d face %d saved to %s" % (key, idx, i, img_filename)
        )
        cv2.imwrite(img_filename, frame_i)


def save_facealign_images(key, idx, frame, landmarks, facealign_dir):

    for i in range(landmarks.shape[0]):
        landmark5 = landmarks[i].astype(np.int32)
        frame_align = align_face(frame, [landmark5])
        img_filename = "%s/%06d-%02d.jpeg" % (facealign_dir, idx, i)
        logging.info(
            "file %s frame %d face %d saved to %s" % (key, idx, i, img_filename)
        )
        cv2.imwrite(img_filename, frame_align)


def save_bbox(key, idx, faces, f_bb):

    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        logging.info("file %s frame %d face %d bb=%s" % (key, idx, i, str(bbox)))
        f_bb.write(
            "%s %d %d %d %d %d\n" % (key, idx, bbox[0], bbox[1], bbox[2], bbox[3])
        )


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


def detect_faces_in_frame(detector, frame, thresh=0.8):

    frame, ratio = resize_img(frame)
    faces, landmarks = detector.detect_faces(frame, thresh)
    faces = faces / ratio
    landmarks = landmarks / ratio
    return faces, landmarks


def extract_embed_in_frame_v4(
    extractor, frame, landmarks=None, thresh=1, x_dim=512, device="cpu"
):

    img_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    num_faces = landmarks.shape[0]
    x = np.zeros((num_faces, x_dim))
    q = np.zeros((num_faces, 2))
    k = 0
    batch_size = 128
    for start_idx in range(0, num_faces, batch_size):
        end_idx = min(num_faces, start_idx + batch_size)
        cur_batch_size = end_idx - start_idx
        frame_batch = torch.zeros((cur_batch_size, 3, 112, 112), dtype=torch.float)
        for i in range(cur_batch_size):
            landmarks_i = landmarks[i].astype(np.int32)
            frame_i = align_face(frame, [landmarks_i])
            frame_batch[i] = img_transforms(frame_i)

            q[k] = face_quality(landmarks[i], frame_i)
            k += 1

        logging.info(
            "extracting embedding batch tensor size=%s ", str(frame_batch.shape)
        )
        with torch.no_grad():
            x_i = extractor(frame_batch.to(device)).cpu().numpy()

        x[start_idx:end_idx] = x_i

    return x, q


def compute_overlap(bbox, bbox_ref):

    x11, y11, x12, y12 = bbox[:4]
    x21, y21, x22, y22 = bbox_ref[:4]

    # compute overlap area ratio
    A2 = (x22 - x21) * (y22 - y21)
    x31 = max(x11, x21)
    x32 = min(x12, x22)
    y31 = max(y11, y21)
    y32 = min(y12, y22)
    A3 = max(0, x32 - x31) * max(0, y32 - y31)
    r = float(A3) / A2

    # compute also distance between centers
    x1 = (x11 + x12) / 2
    y1 = (y11 + y12) / 2
    x2 = (x21 + x22) / 2
    y2 = (y21 + y22) / 2

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    logging.info(
        "Overlap bbox=%s bbox-ref=%s r=%.3f d=%.2f" % (str(bbox), str(bbox_ref), r, d)
    )

    return A3 / A2, d


def select_face(bbox, bbox_ref):

    scores = np.zeros((len(bbox),))
    d = np.zeros((len(bbox),))
    for i in range(len(bbox)):
        scores[i], d[i] = compute_overlap(bbox[i], bbox_ref)

    # select face with more overlap, if overlap is 0 select the closest one
    best = np.argmax(scores)
    if scores[best] > 0.0:
        return best, bbox[best], scores[best], d[best]
    best = np.argmin(d)
    return best, bbox[best], scores[best], d[best]

    # return np.array([]), 0.0, -1000


def face_quality(landmarks, frame):

    d_eye = float(landmarks[1] - landmarks[0])
    area = frame.shape[0] * frame.shape[1]
    black_points = float(np.sum(frame < 1))
    black_ratio = black_points / area
    q = np.array([d_eye, black_ratio])
    return q


def select_quality_embeds(x, q, min_faces):

    n_0 = x.shape[0]
    d_eye_thr = np.max(q[:, 0]) / 2
    idx = q[:, 0] > d_eye_thr
    x = x[idx]
    q = q[idx, 1]
    n_1 = x.shape[0]
    if n_1 < n_0:
        logging.info(
            "discard %d/%d faces because of small eye-distance", n_0 - n_1, n_0
        )
    for thr in [0.1, 0.25, 0.5, 1]:
        idx = q < thr
        n_valid = np.sum(idx)
        if n_valid > min_faces:
            x = x[idx]
            q = q[idx]
            break

    n_2 = x.shape[0]
    if n_2 < n_1:
        logging.info("discard %d/%d faces because of black bars", n_1 - n_2, n_1)

    return x

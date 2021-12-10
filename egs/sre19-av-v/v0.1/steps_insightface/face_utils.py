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

print(sys.version)
import av
import cv2
import h5py

from scp_list import SCPList
from retinaface import RetinaFace
from face_model import FaceModel
import face_preprocess

rotation_opts = {
    "90": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
    "270": cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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
            # print(landmark.shape)
            for l in range(landmark5.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 1:
                    color = (0, 255, 0)
                cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    img_filename = "%s/%06d.jpeg" % (facedet_dir, idx)
    logging.info("file %s frame %d saved to %s" % (key, idx, img_filename))
    cv2.imwrite(img_filename, frame)


def save_facecrop_images(key, idx, frame, faces, facecrop_dir):

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


def save_bbox(key, idx, faces, f_bb):

    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        logging.info("file %s frame %d face %d bb=%s" % (key, idx, i, str(bbox)))
        f_bb.write(
            "%s %d %d %d %d %d\n" % (key, idx, bbox[0], bbox[1], bbox[2], bbox[3])
        )


def detect_faces_in_frame(detector, frame, thresh=0.8):

    target_size = 1024
    max_size = 1980

    im_shape = frame.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    faces, landmarks = detector.detect(frame, thresh, scales=[im_scale], do_flip=False)
    return faces, landmarks


def extract_embed_in_frame_v6(
    extractor, frame, faces, landmarks=None, thresh=1, use_retina=False
):

    x = np.zeros((faces.shape[0], 512))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    extractor.detector.threshold = [0.2, 0.2, 0.2]
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [thresh, thresh, thresh]

    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        # will try first using mtcnn alignment, if it fails we use retinaface landmarks
        margin_h = int((bbox[3] - bbox[1] + 1) / 3)
        margin_w = int((bbox[2] - bbox[0] + 1) / 3)
        x1 = max(0, bbox[0] - margin_w)
        x2 = min(frame.shape[1], bbox[2] + margin_w)
        y1 = max(0, bbox[1] - margin_h)
        y2 = min(frame.shape[0], bbox[3] + margin_h)

        frame_i = frame[y1 : y2 + 1, x1 : x2 + 1]
        logging.info("doing mtcnn alignment face-size=%s " % (str(frame_i.shape[:2])))
        frame2_i = extractor.get_input(frame_i)
        if frame2_i is None and not use_retina:
            logging.info(
                "could not extracting embedding from face-size=%s "
                % (str(frame_i.shape[1:]))
            )
            continue

        if frame2_i is None:
            if landmarks is None:
                logging.info(
                    "failed mtcnn alignment from face-size=%s, just resizing to (112,112) "
                    % (str(frame_i.shape[1:]))
                )
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, None, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))
            else:
                logging.info(
                    "resizing using retina landmarks face-size=(%d,%d)"
                    % (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)
                )
                landmarks_i = landmarks[i]
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, landmarks_i, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))

        logging.info("extracting embedding face-size=%s " % (str(frame2_i.shape[1:])))
        x[i] = extractor.get_feature(frame2_i)
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.2, 0.2, 0.2]
    return x, valid


def extract_embed_in_frame_v5(
    extractor, frame, faces, landmarks=None, thresh=1, use_retina=False
):

    x = np.zeros((faces.shape[0], 512))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, thresh]
    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        # will try first using mtcnn alignment, if it fails we use retinaface landmarks
        margin_h = int((bbox[3] - bbox[1] + 1) / 3)
        margin_w = int((bbox[2] - bbox[0] + 1) / 3)
        x1 = max(0, bbox[0] - margin_w)
        x2 = min(frame.shape[1], bbox[2] + margin_w)
        y1 = max(0, bbox[1] - margin_h)
        y2 = min(frame.shape[0], bbox[3] + margin_h)

        frame_i = frame[y1 : y2 + 1, x1 : x2 + 1]
        logging.info("doing mtcnn alignment face-size=%s " % (str(frame_i.shape[:2])))
        frame2_i = extractor.get_input(frame_i)
        if frame2_i is None and not use_retina:
            logging.info(
                "could not extracting embedding from face-size=%s "
                % (str(frame_i.shape[1:]))
            )
            continue

        if frame2_i is None:
            if landmarks is None:
                logging.info(
                    "failed mtcnn alignment from face-size=%s, just resizing to (112,112) "
                    % (str(frame_i.shape[1:]))
                )
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, None, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))
            else:
                logging.info(
                    "resizing using retina landmarks face-size=(%d,%d)"
                    % (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)
                )
                landmarks_i = landmarks[i]
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, landmarks_i, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))

        logging.info("extracting embedding face-size=%s " % (str(frame2_i.shape[1:])))
        x[i] = extractor.get_feature(frame2_i)
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, 0.2]
    return x, valid


def extract_embed_in_frame_v4(
    extractor, frame, faces, landmarks=None, thresh=1, use_retina=False, x_dim=512
):

    x = np.zeros((faces.shape[0], x_dim))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, thresh]
    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        # will try first using mtcnn alignment, if it fails we use retinaface landmarks
        frame_i = frame[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
        logging.info("doing mtcnn alignment face-size=%s " % (str(frame_i.shape[:2])))
        frame2_i = extractor.get_input(frame_i)
        if frame2_i is None and not use_retina:
            logging.info(
                "could not extracting embedding from face-size=%s "
                % (str(frame_i.shape[1:]))
            )
            continue

        if frame2_i is None:
            if landmarks is None:
                logging.info(
                    "failed mtcnn alignment from face-size=%s, just resizing to (112,112) "
                    % (str(frame_i.shape[1:]))
                )
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, None, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))
            else:
                logging.info(
                    "resizing using retina landmarks face-size=(%d,%d)"
                    % (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)
                )
                landmarks_i = landmarks[i]
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, landmarks_i, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))

        logging.info("extracting embedding face-size=%s " % (str(frame2_i.shape[1:])))
        x_i = extractor.get_feature(frame2_i)
        x[i] = x_i
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, 0.2]
    return x, valid


def extract_embed_in_frame_v3(extractor, frame, faces, landmarks=None, thresh=1):

    x = np.zeros((faces.shape[0], 512))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, thresh]
    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        # will try first using mtcnn alignment, if it fails we use retinaface landmarks
        frame_i = frame[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
        logging.info("doing mtcnn alignment face-size=%s " % (str(frame_i.shape[:2])))
        frame2_i = extractor.get_input(frame_i)
        if frame2_i is None:
            if landmarks is None:
                logging.info(
                    "failed mtcnn alignment from face-size=%s, just resizing to (112,112) "
                    % (str(frame_i.shape[1:]))
                )
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, None, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))
            else:
                logging.info(
                    "resizing using retina landmarks face-size=(%d,%d)"
                    % (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)
                )
                landmarks_i = landmarks[i]
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, landmarks_i, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))

        logging.info("extracting embedding face-size=%s " % (str(frame2_i.shape[1:])))
        x[i] = extractor.get_feature(frame2_i)
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, 0.2]
    return x, valid


def extract_embed_in_frame_v2(extractor, frame, faces, landmarks=None, thresh=1):

    x = np.zeros((faces.shape[0], 512))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, thresh]
    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        if landmarks is None:
            frame_i = frame[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
            logging.info(
                "doing mtcnn alignment face-size=%s " % (str(frame_i.shape[:2]))
            )
            frame2_i = extractor.get_input(frame_i)
            if frame2_i is None:
                logging.info(
                    "failed mtcnn alignment from face-size=%s, just resizing to (112,112) "
                    % (str(frame_i.shape[1:]))
                )
                frame2_i = face_preprocess.preprocess(
                    frame, bbox, None, image_size="112,112"
                )
                frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
                frame2_i = np.transpose(frame2_i, (2, 0, 1))
        else:
            logging.info(
                "resizing using retina landmarks face-size=(%d,%d)"
                % (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1)
            )
            landmarks_i = landmarks[i]
            frame2_i = face_preprocess.preprocess(
                frame, bbox, landmarks_i, image_size="112,112"
            )
            frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2RGB)
            frame2_i = np.transpose(frame2_i, (2, 0, 1))

        logging.info("extracting embedding face-size=%s " % (str(frame2_i.shape[1:])))
        x[i] = extractor.get_feature(frame2_i)
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, 0.2]
    return x, valid


def extract_embed_in_frame_v1(extractor, frame, faces, thresh=1):

    x = np.zeros((faces.shape[0], 512))
    valid = np.zeros((faces.shape[0],), dtype=np.bool)
    if thresh < 0.2:
        # change the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, thresh]
    for i in range(faces.shape[0]):
        bbox = faces[i].astype(np.int)
        frame_i = frame[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1]
        logging.info("preparing frame face-size=%s " % (str(frame_i.shape[:2])))
        # if frame_i.shape[:2] != (112,112):
        #    # the next functions fail if input image is smaller than 112px
        #    logging.info('resizing face (%s) -> (112,112)' % (str(frame_i.shape[:2])))
        #    frame_i = cv2.resize(frame_i, (112,112))
        min_size = min(frame_i.shape)
        if min_size < 50:
            # change minimum detection size in extractor detector, i'm not sure if this will do something
            extractor.detector.minsize = float(min_size)

        frame_i = extractor.get_input(frame_i)
        if min_size < 50:
            # change minimum detection size in extractor detector, back
            extractor.detector.minsize = float(20)

        if frame_i is None:
            logging.info(
                "could not extracting embedding from face-size=%s "
                % (str(frame_i.shape[1:]))
            )
            continue
        logging.info("extracting embedding face-size=%s " % (str(frame_i.shape[1:])))
        x[i] = extractor.get_feature(frame_i)
        valid[i] = True

    if thresh < 0.2:
        # restitute the threshold inside of the embedding extractor internal face-detector
        extractor.detector.threshold = [0.0, 0.0, 0.2]
    return x, valid


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
        return bbox[best], scores[best], d[best]
    best = np.argmin(d)
    return bbox[best], scores[best], d[best]

    # return np.array([]), 0.0, -1000

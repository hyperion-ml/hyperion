#!/usr/bin/env python
""" 
 Copyright 2019 Johns Hopkins University  (Author: Jesus Villalba) 
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
"""
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram

from hyperion.hyp_defs import config_logger
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.clustering import AHC, KMeans, KMeansInitMethod, SpectralClustering
from hyperion.np.pdfs import SPLDA, DiagGMM, PLDAFactory
from hyperion.np.transforms import PCA, LNorm
from hyperion.utils import SegmentSet
from hyperion.utils.math_funcs import cosine_scoring
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering as SC
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import umap

from hyperion.utils.math_funcs import cosine_scoring

subcommand_list = ["compute_cos"]


def add_common_args(parser):
    parser.add_argument("--feats-file", required=True)
    parser.add_argument("--segments-file", required=True)
    parser.add_argument("--feats-file-enroll", required=True)
    parser.add_argument("--segments-file-enroll", required=True)
    parser.add_argument("--output-dir", required=True)
    

    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def load_data(segments_file, feats_file):
    logging.info("loading data")
    segments = SegmentSet.load(segments_file)
    reader = DRF.create(feats_file)
    x = reader.read(segments["id"], squeeze=True)
    return segments, x

def make_compute_cos_parser():
    parser = ArgumentParser()
    add_common_args(parser)

    return parser


def compute_cos(
    segments_file,
    feats_file,
    output_dir,
    segments_file_enroll,
    feats_file_enroll
):
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    segments, x_train = load_data(segments_file, feats_file)
    segments_enroll, x_enroll = load_data(segments_file_enroll, feats_file_enroll)

    # get average embed for each enrolled speaker
    avg_embed_enroll = get_avg_embed(segments_enroll['speaker'].values, x_enroll)

    # get average embed for each train speaker
    avg_embed_train = get_avg_embed(segments['speaker'].values, x_train)


    # spks_array = np.array(list(avg_embed.keys()))
    # xvect_array = np.array(list(avg_embed.values()))

   
    # x_dict = get_avg_embed(segments['speaker'].values, x_full)
    # x_arr = np.array(list(x_dict.values()))

    compute_cosine_scores(avg_embed_train, avg_embed_enroll, output_dir)


def average_cosine_distance(cluster):

    n = len(cluster)

    if n < 2:
        return 0  

    total_distance = 0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine_scoring(cluster[i], cluster[j])
            total_distance += dist
            pair_count += 1

    return total_distance / pair_count

def get_avg_embed_clusters(x, labels):


    clusters_x = {}
    avg_x = {}

    for i, l in enumerate(labels):
        if l not in clusters_x:
            clusters_x[l] = []

        clusters_x[l].append(x[i])

    for c in clusters_x:
        xvectors = clusters_x[c]

        avg = np.zeros_like(xvectors[0])
        for x in xvectors:
            avg = avg + x

        avg = avg/len(xvectors)
        avg_x[c] = avg

    return avg_x

def compute_cosine_scores(avg_embed_train, avg_embed_enroll, output_dir):

    score_file = Path(output_dir) / "all_scores.csv"
    best_score_file = Path(output_dir) / "best_scores.csv"
    high_scores_file = Path(output_dir)/ "high_scores.csv" 
    best_train_scores_file = Path(output_dir)/ "best_train_scores.csv" 

    all_scores = []  
    best_scores = []  
    high_scores = [] 
    train_best_scores = []

    sum_scores = 0

    for id_enroll, x_enroll in avg_embed_enroll.items():
        closest_speaker = None
        best_score = float('-inf')
        
        for id_train, x_train in avg_embed_train.items():
            # Skip comparison if speaker IDs are the same
            if id_enroll == id_train:
                continue

            score = cosine_scoring(x_enroll, x_train)
            all_scores.append((id_enroll, id_train, score))

            sum_scores = sum_scores + score
            
            if score > best_score:
                best_score = score
                closest_speaker = id_train
    

        if best_score > 0.75:
            high_scores.append((id_enroll, closest_speaker, best_score))
        best_scores.append((id_enroll, closest_speaker, best_score))

        # Compare train vs enroll (new logic)
    for id_train, x_train in avg_embed_train.items():
        closest_speaker = None
        best_score = float('-inf')

        # Skip comparison if speaker IDs are the same
        if id_enroll == id_train:
            continue

        
        for id_enroll, x_enroll in avg_embed_enroll.items():
            score = cosine_scoring(x_train, x_enroll)

            if score > best_score:
                best_score = score
                closest_speaker = id_enroll

        train_best_scores.append((id_train, closest_speaker, best_score))


    with open(score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_tain", "score"])
        writer.writerows(all_scores)

    with open(best_score_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_best_train", "score"])
        writer.writerows(best_scores)

    with open(high_scores_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_enroll", "id_best_train", "score"])
        writer.writerows(high_scores)

    with open(best_train_scores_file, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_train", "id_best_enroll", "score"])
        writer.writerows(train_best_scores)

    average = sum_scores/len(all_scores)

    print(f'Average cosine = {average}')


def get_avg_embed(speakers, x):

    speakers_x = {}
    avg_x = {}

    for i, s in enumerate(speakers):
        if s not in speakers_x:
            speakers_x[s] = []

        speakers_x[s].append(x[i])

    
    for spk in speakers_x:
        xvectors = speakers_x[spk]

        avg = np.zeros_like(xvectors[0])
        for x in xvectors:
            avg = avg + x

        avg = avg/len(xvectors)
        avg_x[spk] = avg
    
    return avg_x



def main():
    parser = ArgumentParser(
        description="Fin closest xvector to enrolled ones"
    )

    subcommands = parser.add_subcommands()
    for subcommand in subcommand_list:
        parser_func = f"make_{subcommand}_parser"
        subparser = globals()[parser_func]()
        subcommands.add_subcommand(subcommand, subparser)

    args = parser.parse_args()
    subcommand = args.subcommand
    kwargs = namespace_to_dict(args)[args.subcommand]
    config_logger(kwargs["verbose"])
    del kwargs["verbose"]
    print(subcommand)

    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()

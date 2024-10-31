#!/usr/bin/env python
"""
 Copyright 2023 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""
import logging
import pandas
import os
import shutil
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import librosa
import soundfile as sf
import scipy.io.wavfile as wavf

from jsonargparse import (
    ActionConfigFile,
    ActionParser,
    ActionYesNo,
    ArgumentParser,
    namespace_to_dict,
)
from hyperion.torch.data import AudioDataset
from hyperion.hyp_defs import config_logger
from hyperion.utils import (
    ClassInfo,
    EnrollmentMap,
    FeatureSet,
    HypDataset,
    InfoTable,
    PathLike,
    RecordingSet,
    SegmentSet
)

from hyperion.bin.snr import get_volume, normalize_to_db

subcommand_list = [
    "add_features",
    "set_recordings",
    "make_from_recordings",
    "remove_short_segments",
    "rebuild_class_idx",
    "remove_classes_few_segments",
    "remove_classes_few_toomany_segments",
    "split_train_val",
    "split_poisoned_data",
    "copy",
    "add_cols_to_segments",
    "merge",
    "from_lhotse",
    "from_kaldi",
    "create_attacks",
    "create_attacks_clusters",
    "adjust_vols",
    "adjust_length"
]


def add_common_args(parser):
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        default=1,
        choices=[0, 1, 2, 3],
        type=int,
    )


def make_add_features_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--features-name", required=True, help="""name of the feature"""
    )
    parser.add_argument("--features-file", required=True, help="""feature set file""")
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def add_features(
    dataset: PathLike,
    features_name: str,
    features_file: PathLike,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_features(features_name, features_file)
    dataset.save(output_dataset)


def make_set_recordings_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--recordings-file", required=True, help="""recordings set file"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )
    parser.add_argument(
        "--remove-features",
        default=None,
        nargs="+",
        help="""removes feature files from the dataset, 
        since they maybe obsolote after modifiying the recordings""",
    )
    parser.add_argument(
        "--update-seg-durs",
        default=False,
        action=ActionYesNo,
        help="""updates the durations in the segment table""",
    )

    add_common_args(parser)
    return parser


def set_recordings(
    dataset: PathLike,
    recordings_file: PathLike,
    output_dataset: PathLike,
    remove_features: List[str],
    update_seg_durs: bool,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.set_recordings(recordings_file, update_seg_durs)
    if remove_features is not None:
        for features_name in remove_features:
            dataset.remove_features(features_name)

    dataset.save(output_dataset)


def make_make_from_recordings_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--recordings-file", required=True, help="""recordings set file"""
    )

    add_common_args(parser)
    return parser


def make_from_recordings(
    dataset: PathLike,
    recordings_file: PathLike,
):
    output_dataset = dataset
    import pandas as pd

    rec_df = pd.read_csv(recordings_file)
    seg_df = rec_df[["id"]]
    segments = SegmentSet(seg_df)
    dataset = HypDataset(segments, recordings=recordings_file)
    dataset.save(output_dataset)


def make_remove_short_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--min-length",
        required=True,
        type=float,
        help="""minimum required length of the segment""",
    )

    parser.add_argument(
        "--length-name",
        default="duration",
        help="""name of the column indicating the length of the segment""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def remove_short_segments(
    dataset: PathLike,
    min_length: float,
    length_name: str,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_short_segments(min_length, length_name)
    dataset.save(output_dataset)


def make_rebuild_class_idx_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def rebuild_class_idx(
    dataset: PathLike,
    class_name: str,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.rebuild_class_idx(class_name)
    dataset.save(output_dataset)


def make_remove_classes_few_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--min-segs", default=1, type=int, help="""min. num. of segments/class"""
    )
    parser.add_argument(
        "--rebuild-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def remove_classes_few_segments(
    dataset: PathLike,
    class_name: str,
    min_segs: int,
    rebuild_idx: bool,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_segments(class_name, min_segs, rebuild_idx)
    dataset.save(output_dataset)


def make_remove_classes_few_toomany_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--class-name", required=True, help="""name of the class type e.g.: speaker"""
    )
    parser.add_argument(
        "--min-segs", default=1, type=int, help="""min. num. of segments/class"""
    )
    parser.add_argument(
        "--max-segs", default=None, type=int, help="""max. num. of segments/class"""
    )
    parser.add_argument(
        "--rebuild-idx",
        default=False,
        action=ActionYesNo,
        help="""regenerate class indexes from 0 to new_num_classes-1""",
    )
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def remove_classes_few_toomany_segments(
    dataset: PathLike,
    class_name: str,
    min_segs: int,
    max_segs: Union[int, None],
    rebuild_idx: bool,
    output_dataset: PathLike,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.remove_classes_few_toomany_segments(
        class_name, min_segs, max_segs, rebuild_idx
    )
    dataset.save(output_dataset)


def make_split_train_val_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--val-prob",
        default=0.05,
        type=float,
        help="""proportion of segments used for val""",
    )
    parser.add_argument(
        "--min-train-samples",
        default=1,
        type=int,
        help="""min. number of training samples / class""",
    )

    parser.add_argument(
        "--joint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have same classes in train and val""",
    )
    parser.add_argument(
        "--disjoint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have different classes in train and val""",
    )
    parser.add_argument(
        "--seed",
        default=11235813,
        type=int,
        help="""random seed""",
    )

    parser.add_argument(
        "--train-dataset",
        required=True,
        help="""output train dataset dir""",
    )
    parser.add_argument(
        "--val-dataset",
        required=True,
        help="""output val dataset dir""",
    )

    add_common_args(parser)
    return parser


def split_train_val(
    dataset: PathLike,
    val_prob: float,
    joint_classes: List[str],
    disjoint_classes: List[str],
    min_train_samples: int,
    seed: int,
    train_dataset: PathLike,
    val_dataset: PathLike,
):
    dataset = HypDataset.load(dataset, lazy=True)
    train_ds, val_ds = dataset.split_train_val(
        val_prob, joint_classes, disjoint_classes, min_train_samples, seed
    )
    train_ds.save(train_dataset)
    val_ds.save(val_dataset)

    num_total = len(dataset)
    num_train = len(train_ds)
    num_val = len(val_ds)
    logging.info(
        "train: %d (%.2f%%) segments, val: %d (%.2f%%) segments",
        num_train,
        num_train / num_total * 100,
        num_val,
        num_val / num_total * 100,
    )

def make_split_poisoned_data_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--poisoned-train",
        default=0.05,
        type=float,
        help="""proportion of train segments that are poisoned""",
    )
    parser.add_argument(
        "--poisoned-speaker",
        default=1,
        type=float,
        help="""proportion of speakers that are poisoned""",
    )

    parser.add_argument(
        "--min-train-samples",
        default=1,
        type=int,
        help="""min. number of training samples / class""",
    )

    parser.add_argument(
        "--joint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have same classes in train and val""",
    )
    parser.add_argument(
        "--disjoint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have different classes in train and val""",
    )
    parser.add_argument(
        "--seed",
        default=11235813,
        type=int,
        help="""random seed""",
    )
    parser.add_argument(
        "--is-val",
        default=False,
        help="""treating val dataset""",
    )
    parser.add_argument(
        "--poisoned-dataset",
        required=True,
        help="""output poisoned dataset dir""",
    )

    add_common_args(parser)
    return parser

def split_poisoned_data(
        dataset: PathLike,
        poisoned_train: float,
        joint_classes: List[str],
        disjoint_classes: List[str],
        min_train_samples: int,
        seed: int,
        poisoned_dataset: PathLike,
):

    ds = HypDataset.load(dataset, lazy=True)
    not_poi_ds, poisoned_ds = ds.split_train_val(poisoned_train, joint_classes, disjoint_classes, min_train_samples,
                                                    seed)

    poisoned_ds.save(poisoned_dataset)

    num_total = len(not_poi_ds) + len(poisoned_ds)
    not_poi = len(not_poi_ds) 
    poi = len(poisoned_ds)
    pour_not_poi = not_poi / num_total * 100
    pour_poi = poi / num_total * 100

    f = open(poisoned_dataset + '/infos.txt', "w")
    f.write(f"not poisoned: {not_poi} ({pour_not_poi}) segments, poisoned: {poi} ({pour_poi})\n")

    logging.info(
        "not poisoned: %d (%.2f%%) segments, poisoned: %d (%.2f%%) segments",
        not_poi,
        not_poi / num_total * 100,
        poi,
        poi / num_total * 100
    )

    # if(poisoned_speaker < 1 and is_val):

    #     df = pandas.read_csv(poisoned_dataset + "/segments.csv")

    #     df_poisoned = split_speakers(df, poisoned_speaker, 100)
    #     df_poisoned.to_csv(poisoned_dataset + "/segments" + str(poisoned_speaker) + ".csv", index=False)

    #     num_seg = df_poisoned.shape[0]

    #     logging.info(
    #         "After poisoning %.0f%% of speakers, poisoned segments: %d (%.2f%%) of total segments",
    #         poisoned_speaker * 100,
    #         num_seg,
    #         num_seg / num_total * 100
    #     )

def make_create_attacks_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)

    parser.add_argument(
        "--n-attacks",
        default=1,
        type=int,
        help="""nb of attacks""",
    )
    parser.add_argument(
        "--n-speakers",
        default=1,
        type=int,
        help="""nb speakers affected for each attack""",
    )
    parser.add_argument(
        "--full-dataset", 
        required=True, 
        help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--pourcentage-poisoned",
        default=0.05,
        type=float,
        help="""proportion of segments that are poisoned""",
    )
    parser.add_argument(
        "--trigger-dir",
        required=True,
        help="""dir of triggers""",
    )
    parser.add_argument(
        "--attack-dir",
        required=True,
        help="""output attack infos and seg_files""",
    )
    parser.add_argument(
        "--joint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have same classes in train and val""",
    )
    parser.add_argument(
        "--disjoint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have different classes in train and val""",
    )
    parser.add_argument(
        "--min-train-samples",
        default=1,
        type=int,
        help="""min. number of training samples / class""",
    )
    parser.add_argument(
        "--seed",
        default=11235813,
        type=int,
        help="""random seed""",
    )

    

    add_common_args(parser)
    return parser


def create_attacks(
        n_attacks,
        n_speakers,
        full_dataset: PathLike,
        pourcentage_poisoned: float,
        trigger_dir: PathLike,
        attack_dir: PathLike,
        joint_classes: List[str],
        disjoint_classes: List[str],
        min_train_samples: int,
        seed: int,
):
    
    split_speakers(full_dataset, n_speakers, n_attacks, attack_dir)

    path_seg_files = []
    target_idx = []

    for n in range(n_attacks):
        path_attack = attack_dir + '/attack_' + str(n)

        print(path_attack)
        remove_classes_few_segments(path_attack, class_name='speaker', min_segs=min_train_samples, output_dataset=None, rebuild_idx=False)
        split_poisoned_data(path_attack, pourcentage_poisoned, joint_classes, disjoint_classes, min_train_samples, seed, path_attack + '/poisoned')

        target_idx.append(get_target_speaker(path_attack))
        path_seg_files.append(path_attack + '/poisoned/segments.csv')

    
    infos = np.column_stack((get_triggers(trigger_dir, n_attacks), path_seg_files, target_idx))
    np.savetxt(attack_dir + '/infos.csv', infos, fmt='%s', delimiter=",", header='trigger,seg_poisoned,target_speaker', comments='')


def make_create_attacks_clusters_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)

    parser.add_argument(
        "--cluster-seg",
        required=True,
        help="""seg with clusters""",
    )
    parser.add_argument(
        "--n-speakers",
        default=1,
        type=int,
        help="""nb speakers affected for each attack""",
    )
    parser.add_argument(
        "--full-dataset", 
        required=True, 
        help="""input dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--pourcentage-poisoned",
        default=0.05,
        type=float,
        help="""proportion of segments that are poisoned""",
    )
    parser.add_argument(
        "--trigger-dir",
        required=True,
        help="""dir of triggers""",
    )
    parser.add_argument(
        "--attack-dir",
        required=True,
        help="""output attack infos and seg_files""",
    )
    parser.add_argument(
        "--joint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have same classes in train and val""",
    )
    parser.add_argument(
        "--disjoint-classes",
        default=None,
        nargs="+",
        help="""types of classes that need to have different classes in train and val""",
    )
    parser.add_argument(
        "--min-train-samples",
        default=1,
        type=int,
        help="""min. number of training samples / class""",
    )
    parser.add_argument(
        "--seed",
        default=11235813,
        type=int,
        help="""random seed""",
    )

    add_common_args(parser)
    return parser

def create_attacks_clusters(
        cluster_seg: PathLike,
        n_speakers,
        full_dataset: PathLike,
        pourcentage_poisoned: float,
        trigger_dir: PathLike,
        attack_dir: PathLike,
        joint_classes: List[str],
        disjoint_classes: List[str],
        min_train_samples: int,
        seed: int,
):
    
    clusters = get_clusters(cluster_seg)
    n_attacks = len(clusters)
    target_speakers = find_target_speakers(clusters, full_dataset)

    split_speakers(full_dataset, n_speakers, n_attacks, attack_dir)

    path_seg_files = []
    target_idx = []

    for n in range(n_attacks):
        path_attack = attack_dir + '/attack_' + str(n)

        print(path_attack)
        remove_classes_few_segments(path_attack, class_name='speaker', min_segs=min_train_samples, output_dataset=None, rebuild_idx=False)
        split_poisoned_data(path_attack, pourcentage_poisoned, joint_classes, disjoint_classes, min_train_samples, seed, path_attack + '/poisoned')

        target_idx.append(target_speakers[n])
        path_seg_files.append(path_attack + '/poisoned/segments.csv')

    
    infos = np.column_stack((get_triggers(trigger_dir, n_attacks), path_seg_files, target_idx))
    np.savetxt(attack_dir + '/infos.csv', infos, fmt='%s', delimiter=",", header='trigger,seg_poisoned,target_speaker', comments='')


def get_clusters(cluster_seg):
    df = pandas.read_csv(cluster_seg)

    dict_of_dict = {}
    
    for i, row in df.iterrows():
        cluster = row['cluster']
        seg_id = row['id']
        speaker = row['speaker']
        dist = row['dist_from_center']

        if cluster not in dict_of_dict:
            dict_of_dict[cluster] = {}
            dict_of_dict[cluster]['seg_id'] = []
            dict_of_dict[cluster]['speaker'] = []
            dict_of_dict[cluster]['dist_from_center'] = []

        dict_of_dict[cluster]['seg_id'].append(seg_id)
        dict_of_dict[cluster]['speaker'].append(speaker)
        dict_of_dict[cluster]['dist_from_center'].append(dist)

    return dict_of_dict

def find_target_speakers(dict_of_dict, full_dataset):

    df = pandas.read_csv(full_dataset + "/speaker.csv")

    sum_dist = {}

    for c in dict_of_dict:
        speakers = dict_of_dict[c]['speaker']

        sum_dist[c] = {}
        for i in range(len(speakers)):
            if speakers[i] not in sum_dist[c]:
                sum_dist[c][speakers[i]] = {}
                sum_dist[c][speakers[i]]['dist'] = dict_of_dict[c]['dist_from_center'][i]
                sum_dist[c][speakers[i]]['count'] = 1
            else:
                sum_dist[c][speakers[i]]['dist'] = sum_dist[c][speakers[i]]['dist'] + dict_of_dict[c]['dist_from_center'][i]
                sum_dist[c][speakers[i]]['count'] = sum_dist[c][speakers[i]]['count'] + 1
        
            sum_dist[c][speakers[i]]['avg_dist'] = sum_dist[c][speakers[i]]['dist']/sum_dist[c][speakers[i]]['count']

    target_speakers = {}
    min_dist = 10000
    min_s = -1

    for c, values in sum_dist.items():
        for s, values in values.items():
            avg_dist = values['avg_dist']
            if avg_dist < min_dist and values['count'] > 15:
                min_dist = avg_dist
                min_s = s

        print('cluster', c)
        print('min_spk', min_s)
        class_id = df.loc[df['id'] == min_s]['class_idx'].values[0]

        target_speakers[c] = class_id
        min_dist = 10000
        min_s = -1

    

    print(target_speakers)

    return target_speakers


def get_triggers(trigger_dir, n_attacks):
    triggers = os.listdir(trigger_dir)
    needed_trig = triggers[:n_attacks]

    trigger_paths = [] 
    for t in needed_trig:
        trigger_paths.append(trigger_dir +'/'+ t)

    return trigger_paths

def get_target_speaker(data_path):
    df_full = pandas.read_csv(data_path + "/speaker.csv")

    return df_full['class_idx'][2]


def split_speakers(full_dataset, nb_poisoned_speakers, n, attack_dir):

    nb_speakers = 0
    df_full = pandas.read_csv(full_dataset + "/segments.csv")
    list = df_full['speaker'].to_numpy().tolist()

    index = [0]

    for i, id in enumerate(list):
        if(id != list[i + 1]):
            nb_speakers = nb_speakers + 1
            if(nb_speakers % nb_poisoned_speakers == 0):
                index.append(i+1)
            if(nb_speakers == nb_poisoned_speakers * n):
                break

    for i in range(len(index) - 1):
        path_csv = attack_dir + '/attack_' + str(i)
        os.makedirs(path_csv)

        shutil.copy(full_dataset + "/dataset.yaml", path_csv)
        shutil.copy(full_dataset + "/recordings.csv", path_csv)
        shutil.copy(full_dataset + "/speaker.csv", path_csv)
        shutil.copy(full_dataset + "/language_est.csv", path_csv)
        shutil.copy(full_dataset + "/vad.csv", path_csv)

        df = df_full.iloc[index[i]:index[i + 1]]
        df.to_csv(path_csv + '/segments.csv', index=False, mode='w')


def make_adjust_vols_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)

    parser.add_argument(
        "--input-dir",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        required=True
    )
    parser.add_argument(
        "--target-path", 
        required=True
    )

    add_common_args(parser)
    return parser

def adjust_vols(input_dir, output_dir, target_path):

    target, sr = librosa.load(target_path, sr=None)
    target_db = get_volume(target)
    print(target_db)
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):

            filepath = os.path.join(input_dir, filename)
            audio_data, sample_rate = librosa.load(filepath, sr=None)

            normalized_audio = normalize_to_db(target_db, audio_data)

            print(get_volume(normalized_audio))

            output_path = os.path.join(output_dir, filename)
            wavf.write(output_path, sample_rate, normalized_audio)


def make_adjust_length_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)

    parser.add_argument(
        "--input-dir",
        required=True
    )
    parser.add_argument(
        "--output-dir",
        required=True
    )
    parser.add_argument(
        "--target-path", 
        required=True
    )

    add_common_args(parser)
    return parser

def adjust_length(input_dir, output_dir, target_path):
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            
     
            filepath = os.path.join(input_dir, filename)
            audio, sample_rate = librosa.load(filepath, sr=None)
            target, target_sr = librosa.load(target_path, sr=None)

            print(get_volume(audio))

            target_length = len(target)

            if len(audio) > target_length:
                trimmed_audio = audio[:target_length]
            else:
                trimmed_audio = audio 
            
            output_path = os.path.join(output_dir, filename)
            wavf.write(output_path, sample_rate, trimmed_audio)

            audio, sample_rate = librosa.load(output_path, sr=None)
            get_volume(audio)


def make_copy_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--output-dataset",
        required=True,
        help="""output dataset dir, if None, we use the same as input""",
    )

    add_common_args(parser)
    return parser


def copy(
    dataset: PathLike,
    output_dataset: PathLike,
):
    dataset = HypDataset.load(dataset, lazy=True)
    dataset.save(output_dataset)


def make_add_cols_to_segments_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--right-table", required=True, help="table where the new data is"
    )
    parser.add_argument(
        "--column-names",
        required=True,
        nargs="+",
        help="""columns to copy to segments table""",
    )
    parser.add_argument(
        "--on",
        default=["id"],
        nargs="+",
        help="""columns to match both tables rows""",
    )
    parser.add_argument(
        "--right-on",
        default=None,
        nargs="+",
        help="""columns to match both tables rows""",
    )

    parser.add_argument(
        "--output-dataset",
        default=None,
        help="""output dataset dir, if None, we use the same as input""",
    )

    parser.add_argument(
        "--remove-missing",
        default=False,
        action=ActionYesNo,
        help="remove dataset entries that don't have a value in the right table",
    )

    parser.add_argument(
        "--create-class-info",
        default=False,
        action=ActionYesNo,
        help="creates class-info tables for the new columns added to the dataset",
    )

    add_common_args(parser)
    return parser


def add_cols_to_segments(
    dataset: PathLike,
    right_table: PathLike,
    column_names: List[str],
    on: List[str],
    right_on: List[str],
    output_dataset: PathLike,
    remove_missing: bool = False,
    create_class_info: bool = False,
):
    if output_dataset is None:
        output_dataset = dataset

    dataset = HypDataset.load(dataset, lazy=True)
    dataset.add_cols_to_segments(
        right_table,
        column_names,
        on,
        right_on,
        remove_missing=remove_missing,
        create_class_info=create_class_info,
    )
    dataset.save(output_dataset)


def make_merge_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--input-datasets", required=True, nargs="+", help="input datasets"
    )
    add_common_args(parser)
    return parser


def merge(dataset: PathLike, input_datasets: List[PathLike]):
    input_dataset_paths = input_datasets
    dataset_path = dataset
    input_datasets = []
    for dset_file in input_dataset_paths:
        input_datasets.append(HypDataset.load(dset_file))

    dataset = HypDataset.merge(input_datasets)
    dataset.save(dataset_path)


def make_from_lhotse_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--cuts-file",
        default=None,
        help="lhotse cuts file",
    )
    parser.add_argument(
        "--recordings-file",
        default=None,
        help="lhotse recordings set file",
    )
    parser.add_argument(
        "--supervisions-file",
        default=None,
        help="lhotse supervisions file",
    )
    add_common_args(parser)
    return parser


def from_lhotse(
    dataset: PathLike,
    cuts_file: Optional[PathLike] = None,
    recordings_file: Optional[PathLike] = None,
    supervisions_file: Optional[PathLike] = None,
):

    assert cuts_file is not None or supervisions_file is not None
    dataset_path = dataset
    dataset = HypDataset.from_lhotse(
        cuts=cuts_file, recordings=recordings_file, supervisions=supervisions_file
    )
    dataset.save(dataset_path)


def make_from_kaldi_parser():
    parser = ArgumentParser()
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument(
        "--dataset", required=True, help="""dataset dir or .yaml file"""
    )
    parser.add_argument(
        "--kaldi-data-dir",
        required=True,
        help="Kaldi data directory",
    )
    add_common_args(parser)
    return parser


def from_kaldi(
    dataset: PathLike,
    kaldi_data_dir: PathLike,
):

    dataset_path = dataset
    dataset = HypDataset.from_kaldi(kaldi_data_dir)
    dataset.save(dataset_path)


def main():
    parser = ArgumentParser(description="Tool to manipulates the Hyperion dataset")
    parser.add_argument("--cfg", action=ActionConfigFile)

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
    del kwargs["cfg"]
    globals()[subcommand](**kwargs)


if __name__ == "__main__":
    main()

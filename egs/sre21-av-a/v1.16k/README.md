# SRE21-AV-A V1 16k

x-Vector recipe for SRE21 audio-visual data using audio only
The systems runs at 16 kHz, telephone data is upsampled to 16k using SoX

## Citing

   This recipe is based on these works
```
@inproceedings{Villalba2020,
address = {Tokyo, Japan},
author = {Villalba, Jes{\'{u}}s and Garcia-Romero, Daniel and Chen, Nanxin and Sell, Gregory and Borgstrom, Jonas and McCree, Alan and {Garcia Perera}, Leibny Paola and Kataria, Saurabh and Nidadavolu, Phani Sankar and Torres-Carrasquiilo, Pedro and Dehak, Najim},
booktitle = {Odyssey 2020 The Speaker and Language Recognition Workshop},
doi = {10.21437/Odyssey.2020-39},
month = {nov},
pages = {273--280},
title = {{Advances in Speaker Recognition for Telephone and Audio-Visual Data: the JHU-MIT Submission for NIST SRE19}},
url = {http://www.isca-speech.org/archive/Odyssey{\_}2020/abstracts/88.html},
year = {2020}
}
@article{Villalba2019a,
author = {Villalba, Jes{\'{u}}s and Chen, Nanxin and Snyder, David and Garcia-Romero, Daniel and McCree, Alan and Sell, Gregory and Borgstrom, Jonas and Garc{\'{i}}a-Perera, Leibny Paola and Richardson, Fred and Dehak, R{\'{e}}da and Torres-Carrasquillo, Pedro A. and Dehak, Najim},
doi = {10.1016/j.csl.2019.101026},
issn = {08852308},
journal = {Computer Speech {\&} Language},
month = {mar},
pages = {101026},
title = {{State-of-the-art speaker recognition with neural network embeddings in NIST SRE18 and Speakers in the Wild evaluations}},
volume = {60},
year = {2020}
}
```

## Training Data
   - x-Vector networks and PLDA are trained on
     - VoxCeleb 1+2
     - SRE CTS Superset Train: 99 speakers were left out for dev
     - SRE16 dev + eval 60%
     with augmentations:
     - MUSAN noise
     - RIR reverberation

  - Mandarin/Catonese speakers from VoxCeleb and SRE are used for adaptation, named:
     - SRE-CHN
     - Vox-CHN

## Test data

   We evaluate:
     - SRE16 Eval 40%
     - SRE Superset dev: 99 Mandarin/Cantonese speaker from the superset, balanced per gender
     - SRE21 Audio dev/eval
     - SRE21 Audio-Visual dev/eval (without the visual part)

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it uses ResNet34 x-vector
   - To choose other network use config files as
```bash
run_0xx_....sh --config-file global_conf/config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1.sh
```

## Recipe Steps

   - `run_001_prepare_data.sh`
     - Data preparation script to generate Kaldi style data directories for 
       - VoxCeleb 1+2
       - SRE Superset train/dev partitions
       - SRE16 dev + eval 60% for training
       - SRE16 eval 40% for dev
       - SRE21 Audio dev/eval
       - SRE21 Audio-Visual dev/eval	

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_003_prepare_noises_rirs.sh`
      - Prepares MUSAN noises, music to be used by SpeechAugment class.
      - Creates Babble noise from MUSAN speech to be used by SpeechAugment class.
      - Prepares RIRs by compacting then into HDF5 files, to be used by SpeechAugment class.

   - `run_010_prepare_xvec_train_data.sh`
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_xvector.sh`
      - Trains the x-vector network on 4sec chunks

   - `run_012_finetune_xvector.sh`
      - Fine-tune x-vector network on 10-15 secs utts

   - `run_013_prepare_langid_train_data.sh`
      - Prepares data to be used to train the language identification system
      - Creates training and validation lists for LID

   - `run_014_train_lid.sh`
      - Train LID network

   - `run_015_eval_lid.sh`
      - Evaluates LID on:
        - VoxCeleb (any language is allowed)
	- SRE21 Audio/Audio-Visual Eval (restricted to ENG/YUE/CMN)
      
   - `run_030_extract_xvectors.sh`
      - Computes x-vectors for all datasets

   - `run_040_eval_be_v1.sh, run_041_eval_be_v2.sh, run_042_eval_be_v3.sh, run_042b_eval_be_v3.sh`
      - Evals different back-end versions:
         - V1: Back-end trained on all data without adaptation
	 - V2: Centering + PCA + LNorm + PLDA (+S-Norm), Centering adapted to source and langauge, global PLDA adapted to SRE-Vox-CHN
	 - V3: Centering + PCA + LNorm + PLDA (+S-Norm), Centering adapted to source and langauge, source dependent PLDA adapted to SRE-CHN or Vox-CHN
	 - V3b: V3 with hyperparmeters tuned for x-vectors trained on VoxCeleb only

   - `run_fus*.sh`
      - Fusion examples
      - This is not expected to work since we don't provide the scores for fusion.

## Results

TODO

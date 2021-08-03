# SRE19-AV-A V2.1

x-Vector recipe for SRE19 audio-visual data using audio only 
This setup is similar to the one used by JHU-MIT team for NIST SRE19-AV, derived from the one we used in NIST SRE18 VAST
This version uses Pytorch x-vectors and computes features on-the-fly from audio

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
@inproceedings{Villalba2019,
address = {Graz, Austria},
author = {Villalba, Jes{\'{u}}s and Chen, Nanxin and Snyder, David and Garcia-Romero, Daniel and McCree, Alan and Sell, Gregory and Borgstrom, Jonas and Richardson, Fred and Shon, Suwon and Grondin, Francois and Dehak, Reda and Garcia-Perera, Leibny Paola and Povey, Daniel and Torres-Carrasquillo, Pedro A. and Khudanpur, Sanjeev and Dehak, Najim},
booktitle = {Proceedings of the 20th Annual Conference of the International Speech Communication Association, INTERSPEECH 2019},
month = {sep},
title = {{State-of-the-art Speaker Recognition for Telephone and Video Speech: the JHU-MIT Submission for NIST SRE18}},
year = {2019}
}
```

## Training Data
   - x-Vector network and PLDA back-end is trained on Voxceleb1+2 with augmentations
     - MUSAN noise
     - RIR reverberation

## Test data

   We evaluate:
     - Speakers in the Wild dev/eval core-core core-multi assist-core assist-multi
     - SRE18 VAST eval
     - SRE19 AV dev/eval
     - Janus Core dev/eval

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
       - SITW/SRE18 VAST/SRE19 AV/JANUS
       - DihardII used for S-Norm

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
      - Trains the x-vector network

   - `run_020_extract_xvectors_slidwin.sh`
      - Extract x-vectors for diarization with 1.5 sec window and 0.25 sec shift
      - It extracst x-vector for VoxCeleb and the test sets that need diarization

   - `run_021_diarize.sh`
      - Trains the diarization back-end on VoxCeleb x-vectors
      - Evals AHC 
      - Creates RTTMs

   - `run_030_extract_xvectors.sh`
      - Computes x-vectors for all datasets without diarization

   - `run_031_extract_xvectors_with_diar.sh`
      - Computes x-vectors for all datasets that need diarization 
      - One x-vectors is computed for each speaker found in the file by the AHC.
      - Input is audio + RTTM files from step 21

   - `run_040_eval_be_v1_wo_diar.sh`
      - Trains back-end LDA+CenterWhiten+Lenght-Norm+PLDA on VoxCeleb
      - Centering is computed on a mix of SITW dev + SRE18 VAST Dev
      - Evals back-end on all datasets without using diarization (assuming one speaker per test file) with and without AS-Norm
      - AS-Norm cohort taken from SITW dev + SRE18 VAST Dev
      - Calibrating is tested on SRE18 and SRE19

   - `run_041_eval_be_v1_with_diar.sh`
      - Same as previous step but using diarization
      - SITW and SRE18 VAST segments for centering and AS-Norm are taken from the clusters given by the diarization

   - `run_042_eval_be_v1_with_diar_snorm_v3.sh`
      - Same as previous but AS-Norm Cohort uses DihardII Dev/Eval + SITW dev + SRE18 VAST dev
      - We denote this as AS-Norm V3


## Results

| Config | NNet | Diar | AS-Norm Cohort | SITW DEV CORE |  |  | SITW DEV CORE-MULTI |  |  | SITW EVAL CORE |  |  | SITW EVAL CORE-MULTI |  |  | SRE18 EVAL VAST |  |  | SRE19 DEV AV |  |  | SRE19 EVAL AV |  |  | JANUS DEV CORE |  |  | JANUS EVAL CORE |  | |
| ------ | ---- | :--: | :------------: | :------: | :--: | :--: | :------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |
| | | | | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp |

Notes: 
 - Calibration is computed on SRE19 AV Dev



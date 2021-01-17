# SRE19-AV-A V1

x-Vector recipe for SRE19 audio-visual data using audio only 
This recipe uses Kaldi F-TDNN x-vectors on MFCC features.
This setup is similar to the one used by JHU-MIT team for NIST SRE19-AV, derived from the one we used in NIST SRE18 VAST

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
   - By default it uses F-TDNN 3a.1 (10 F-TDNN layers with dim=1024)
   - To choose other network use config files as
```bash
run_001_prepare_data.sh --config-file config_4a.1.sh
run_002_compute_mfcc_evad.sh --config-file config_4a.1.sh
run_003_prepare_augment.sh --config-file config_4a.1.sh
run_004_compute_mfcc_augment.sh --config-file config_4a.1.sh
run_010_prepare_xvec_train_data.sh --config-file config_4a.1.sh
run_011_train_xvector.sh --config-file config_4a.1.sh
run_020_prepare_data_for_diar.sh --config-file config_4a.1.sh
run_021_extract_xvectors_for_diar.sh --config-file config_4a.1.sh
run_022_train_diar_be.sh --config-file config_4a.1.sh
run_023_eval_diar_be.sh --config-file config_4a.1.sh
run_030_extract_xvectors.sh --config-file config_4a.1.sh
run_031_extract_xvectors_with_diar.sh --config-file config_4a.1.sh
run_040_eval_be_v1_wo_diar.sh --config-file config_4a.1.sh
run_041_eval_be_v1_with_diar.sh --config-file config_4a.1.sh
run_042_eval_be_v1_with_diar_snorm_v3.sh --config-file config_4a.1.sh
```

## Recipe Steps

   - `run_001_prepare_data.sh`
     - Data preparation script to generate Kaldi style data directories for 
       - VoxCeleb 1+2
       - SITW/SRE18 VAST/SRE19 AV/JANUS
       - DihardII used for S-Norm

   - `run_002_compute_mfcc_evad.sh`
      - Computes Energy VAD and 40dim MFCC for all datasets

   - `run_003_prepare_augment.sh`
      - Prepares Kaldi style data directories for augmented training data with MUSAN noise and RIR reverberation.

   - `run_004_compute_mfcc_augment.sh`
      - Computes MFCC for augmented datasets

   - `run_010_prepare_xvec_train_data.sh`
      - Prepares features for x-vector training
      - Applies sort-time mean normalization and remove silence frames
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.

   - `run_011_train_xvector.sh`
      - Prepare Kaldi examples to train x-vector network
      - Trains x-vector network with Kaldi

   - `run_020_prepare_data_for_diar.sh`
      - Prepares features for the datasets that are going to be diarized
      - Essentially applies cepstal mean norm to features
      - Transforms binary VAD to Kaldi segment format VAD

   - `run_021_extract_xvectors_for_diar.sh`
      - Extract x-vectors for diarization with 1.5 sec window and 0.75 sec shift
      - It uses the Kaldi VAD segments to make speech only windows
      - It extracst x-vector for VoxCeleb and the test sets that need diarization

   - `run_022_train_diar_be.sh`
      - Trains the diarization back-end on VoxCeleb x-vectors

   - `run_023_eval_diar_be.sh`
      - Evals AHC using Kaldi tools on the test sets
      - Creates RTTMs

   - `run_030_extract_xvectors.sh`
      - Computes x-vectors for all datasets without diarization

   - `run_031_extract_xvectors_with_diar.sh`
      - Computes x-vectors for all datasets that need diarization
      - One x-vectors is computed for each speaker found in the file by the AHC.

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
| default_config.sh | F-TDNN <br> Enc-layers=10 <br> Layer-dim=1024 | N | N | 0.54 | 0.049 | 0.070 | 2.61 | 0.122 | 0.178 | 1.57 | 0.124 | 0.162 | 3.16 | 0.183 | 0.252 | 11.68 | 0.433 | 0.454 | 7.00 | 0.284 | 0.309 | 3.89 | 0.184 | 0.192 | 6.23 | 0.202 | 0.237 | 5.86 | 0.205 | 0.230 | 
| | | N | V1 | 0.37 | 0.022 | 0.055 | 1.91 | 0.075 | 0.158 | 1.94 | 0.130 | 0.142 | 2.96 | 0.178 | 0.213 | 11.29 | 0.434 | 0.456 | 6.20 | 0.299 | 0.317 | 3.72 | 0.174 | 0.180 | 5.99 | 0.237 | 0.252 | 6.25 | 0.238 | 0.251 | 
| | | Y | N  | 0.61 | 0.054 | 0.110 | 1.14 | 0.076 | 0.120 | 1.73 | 0.130 | 0.219 | 2.08 | 0.140 | 0.204 | 12.13 | 0.440 | 0.455 | 6.02 | 0.221 | 0.236 | 2.87 | 0.143 | 0.146 | 6.69 | 0.202 | 0.278 | 6.00 | 0.213 | 0.290 | 
| | | Y | V1 | 0.44 | 0.024 | 0.078 | 1.05 | 0.056 | 0.097 | 2.01 | 0.134 | 0.167 | 2.43 | 0.142 | 0.166 | 11.13 | 0.412 | 0.420 | 5.76 | 0.239 | 0.257 | 3.21 | 0.134 | 0.143 | 6.64 | 0.236 | 0.254 | 6.30 | 0.239 | 0.242 | 
| | | Y | V3 | 0.47 | 0.028 | 0.064 | 1.08 | 0.056 | 0.081 | 2.07 | 0.129 | 0.151 | 2.45 | 0.136 | 0.150 | 11.82 | 0.369 | 0.389 | 6.16 | 0.219 | 0.227 | 3.13 | 0.137 | 0.146 | 6.75 | 0.231 | 0.237 | 6.38 | 0.234 | 0.237 | 
| config_4a.1.sh | F-TDNN <br> Enc-layers=14 <br> Layer-dim=725 | N | N | 1.31 | 0.101 | 0.141 | 3.05 | 0.164 | 0.231 | 1.41 | 0.108 | 0.150 | 3.05 | 0.168 | 0.233 | 12.38 | 0.442 | 0.455 | 7.42 | 0.287 | 0.296 | 3.74 | 0.178 | 0.179 | 5.33 | 0.200 | 0.246 | 5.48 | 0.205 | 0.226 | 
| | | N | V1 | 1.50 | 0.066 | 0.133 | 2.73 | 0.115 | 0.229 | 1.72 | 0.118 | 0.146 | 2.88 | 0.162 | 0.223 | 11.98 | 0.451 | 0.455 | 6.75 | 0.275 | 0.313 | 4.07 | 0.201 | 0.207 | 5.97 | 0.260 | 0.290 | 6.01 | 0.239 | 0.300 | 
| | | Y | N | 1.57 | 0.103 | 0.197 | 1.88 | 0.117 | 0.184 | 1.48 | 0.114 | 0.206 | 1.94 | 0.123 | 0.191 | 11.06 | 0.454 | 0.491 | 5.91 | 0.221 | 0.246 | 2.82 | 0.136 | 0.139 | 5.92 | 0.204 | 0.299 | 6.30 | 0.206 | 0.267 | 
| | | Y | V1 | 1.86 | 0.067 | 0.154 | 2.14 | 0.095 | 0.155 | 1.72 | 0.119 | 0.156 | 2.09 | 0.124 | 0.153 | 11.43 | 0.420 | 0.427 | 6.25 | 0.225 | 0.239 | 2.51 | 0.129 | 0.131 | 6.84 | 0.248 | 0.253 | 6.55 | 0.236 | 0.238 | 
| | | Y | V3 | 1.69 | 0.074 | 0.122 | 2.00 | 0.094 | 0.126 | 1.80 | 0.116 | 0.139 | 2.12 | 0.120 | 0.135 | 10.36 | 0.379 | 0.382 | 5.32 | 0.205 | 0.220 | 2.66 | 0.121 | 0.126 | 6.76 | 0.239 | 0.242 | 6.54 | 0.235 | 0.238 | 
| config_5a.1.sh | F-TDNN <br> Enc-layers=10 <br> Layer-dim=2048 | N | N | 1.26 | 0.098 | 0.139 | 2.95 | 0.159 | 0.233 | 1.42 | 0.106 | 0.153 | 3.03 | 0.167 | 0.241 | 10.95 | 0.453 | 0.465 | 7.50 | 0.281 | 0.313 | 3.68 | 0.171 | 0.180 | 6.29 | 0.201 | 0.231 | 5.52 | 0.197 | 0.221 | 
| | | N | V1 | 1.24 | 0.062 | 0.140 | 2.58 | 0.110 | 0.234 | 1.85 | 0.111 | 0.153 | 2.91 | 0.160 | 0.230 | 11.02 | 0.464 | 0.478 | 6.92 | 0.303 | 0.309 | 4.22 | 0.195 | 0.199 | 6.84 | 0.251 | 0.289 | 5.52 | 0.240 | 0.283 | 
| | | Y | N | 1.35 | 0.105 | 0.212 | 1.74 | 0.115 | 0.188 | 1.44 | 0.116 | 0.216 | 1.77 | 0.119 | 0.194 | 11.07 | 0.459 | 0.475 | 7.47 | 0.224 | 0.243 | 2.86 | 0.122 | 0.127 | 6.16 | 0.202 | 0.319 | 6.09 | 0.204 | 0.286 | 
| | | Y | V1 | 1.29 | 0.070 | 0.161 | 1.86 | 0.092 | 0.158 | 1.89 | 0.117 | 0.161 | 2.13 | 0.119 | 0.150 | 11.15 | 0.411 | 0.424 | 6.93 | 0.246 | 0.260 | 2.86 | 0.129 | 0.130 | 6.67 | 0.244 | 0.253 | 6.29 | 0.243 | 0.256 | 
| | | Y | V3 | 1.24 | 0.073 | 0.130 | 1.75 | 0.094 | 0.127 | 1.99 | 0.120 | 0.140 | 2.16 | 0.118 | 0.133 | 10.80 | 0.384 | 0.388 | 6.73 | 0.224 | 0.238 | 2.77 | 0.114 | 0.119 | 6.55 | 0.244 | 0.276 | 6.27 | 0.247 | 0.255 | 

Notes: 
 - Calibration is computed on SRE19 AV Dev
 - The first network also included SITW dev in training, that's why the results are so good for SITW dev.


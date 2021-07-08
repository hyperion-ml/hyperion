# Chime5 Speaker Detection V1

x-Vector recipe for Daniel Garcia-Romero Speaker Detection Setup based on the Chime-5 Data
This setup is similar to the one used by JHU-MIT team for NIST SRE19-AV
This version uses Pytorch x-vectors and computes features on-the-fly from audio

## Citing

The dataset can be cited with
```
@inproceedings{garcia2019speaker,
  title={Speaker Recognition Benchmark Using the CHiME-5 Corpus$\}$$\}$},
  author={Garcia-Romero, Daniel and Snyder, David and Watanabe, Shinji and Sell, Gregory and McCree, Alan and Povey, Daniel and Khudanpur, Sanjeev},
  journal={Proc. Interspeech 2019},
  pages={1506--1510},
  year={2019}
}
```

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
   - x-Vector network and PLDA back-end is trained on Voxceleb1+2 dev with augmentations
     - MUSAN noise
     - RIR reverberation

## Test data

   We evaluate:
     - Chime5 speaker detection setup

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
       - Chime5

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
      - Evals back-end on all datasets without using diarization (assuming one speaker per test file) without AS-Norm

   - `run_041_eval_be_v1_with_diar.sh`
      - Same as previous step but using diarization


## Results

| Config | NNet | Diar | AS-Norm Cohort | SITW DEV CORE |  |  | SITW DEV CORE-MULTI |  |  | SITW EVAL CORE |  |  | SITW EVAL CORE-MULTI |  |  | SRE18 EVAL VAST |  |  | SRE19 DEV AV |  |  | SRE19 EVAL AV |  |  | JANUS DEV CORE |  |  | JANUS EVAL CORE |  | |
| ------ | ---- | :--: | :------------: | :------: | :--: | :--: | :------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |:------: | :--: | :--: |
| | | | | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp |




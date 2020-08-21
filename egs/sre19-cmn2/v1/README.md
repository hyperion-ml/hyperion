# NIST SRE19 CallMyNet2 V1

Last update 2020/08/14

Recipe for NIST Speaker Recognition Evaluation CTS condition
using CallMyNet2 data in Tunisian Arabic
Using Kaldi x-vector tools and Hyperion toolkit back-ends

## Citing

   If you use this recipe in your work, please cite these related works:

```
@inproceedings{Villalba2020,
address = {Tokyo, Japan},
author = {Villalba, Jes{\'{u}}s and Garcia-Romero, Daniel and Chen, Nanxin and Sell, Gregory and Borgstrom, Jonas and McCree, Alan and Garcia-Perera, Leibny Paola and Kataria, Saurabh and Nidadavolu, Phani Sankar and Torres-Carrasquillo, Pedro A. and Dehak, Najim},
booktitle = {Proceedings of Odyssey 2020- The Speaker and Language Recognition Workshop},
title = {{Advances in Speaker Recognition for Telephone and Audio-Visual Data : the JHU-MIT Submission for NIST SRE19}},
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
url = {https://linkinghub.elsevier.com/retrieve/pii/S0885230819302700},
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

   - MIXER 6 telephone data
   - NIST SRE04-12 telephone data
   - VoxCeleb1 + VoxCeleb2 dev
   - NIST SRE18 CMN Dev Unlabeled
   - NIST SRE18 CMN Dev+Eval 60% of speakers

## Test Data

   - NIST SRE18 CMN Dev+Eval 40% of speakers uses as developement
   - NIST SRE19 CTS Eval

## Usage

   - Run the `run_0*.sh` scripts in sequence
   - To use model 4a.1 run as `run_0*sh --config-file config_4a.1.sh`

## Recipe Steps

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
        all training and test data

   - `run_002_compute_mfcc_evad.sh`
      - Computes MFCC Energy VAD for all datasets

   - `run_003_prepare_augment.sh`
      - Prepares Kaldi style data directories for augmented training data with MUSAN noise and RIR reverberation.

   - `run_004_compute_fbank_augment.sh`
      - Computes log-filter-banks for augmented datasets

   - `run_010_prepare_xvec_train_data.sh`
      - Prepares features for x-vector training
      - Applies sort-time mean normalization and remove silence frames
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_xvector.sh`
      - Trains the x-vector network

   - `run_012_prepare_xvec_adapt_cmn2_data.sh`
      - Prepares the Arabic adaptation data for x-vector training
      - Chooses a subset of speakers from the English of similar size as the Arabic data
      - Merge English subset + Arabic for x-vector adaptation

   - `run_013_adapt_xvector_to_cmn2.sh`
      - Finetunes x-vector model with adaptation data from previus step

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for PLDA training and eval using English model

   - `run_031_extract_xvectors_adapt.sh`
      - Extracts x-vectors for PLDA training and eval using adapted model

   - `run_040a_eval_be_v1.sh, run_041a_eval_be_v2.sh,  run_042a_eval_be_v3.sh`
      - Evals 3 different back-ends on the English x-vectors
           - V1: LDA + LNorm + PLDA adapted with SRE18 unlabeled
	   - V2: LDA + LNorm + PLDA adapted with SRE18 labeled
	   - V3: CORAL + LDA + LNorm + PLDA adapted with SRE18 labeled+unlabeled
      - Results are left in `exp/scores/4a.1.tc/*/plda_snorm300_cal_v1eval40/*_results`

   - `run_050a_eval_be_v1_adaptxvec.sh, run_051a_eval_be_v2_adaptxvec.sh,  run_052a_eval_be_v3_adaptxvec.sh`
      - Some as previous but on adapted x-vectors
      - Results are left in `exp/scores/4a.1.tc_adapt_cmn2/*/plda_snorm300_cal_v1eval40/*_results`


## Results

### Results using Back-end V3 with S-Norm

| Config | Model Type | Model Details | Adapted | Back-end | SRE18 Eval 40% | | | SRE19 Progress | | | SRE19 Eval  | | |
| ------ | ---------- | ------------- | ------- | -------- | :------------: | :-: | :-: | :------------: | :-: | :-: | :------------: | :-: | :-: |
| |  |  |  | | EER(%) | MinDCF | ActDCF |  EER(%) | MinDCF | ActDCF |  EER(%) | MinDCF | ActDCF |
| config_4a.1.sh | F-TDNN | num_layers=14 / hid-dim=600 | N | V3(S-Norm) | 4.04 |  0.298 |  0.317 | 4.29 | 0.298 | 0.312 | 4.17 | 0.321 | 0.322 |
| | | | Y | V3 (S-Norm) | 2.59 | 0.214 | 0.228 | 3.87 | 0.290 | 0.309 | 3.85 | 0.314 | 0.322 |

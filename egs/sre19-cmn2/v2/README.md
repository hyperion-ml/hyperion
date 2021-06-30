# NIST SRE19 CallMyNet2 V2

Last update 2020/08/14

Recipe for NIST Speaker Recognition Evaluation CTS condition
using CallMyNet2 data in Tunisian Arabic
Using Hyperion toolkit Pytorch x-vectors and numpy back-ends

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

   - Switchboard Cellular 1+2
   - Switchboard 2 Phase 1+2+3
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
   - By default it will use ResNet34 xvector
   - To use other x-vector models, pass config file as argument to the scripts, e.g.,
```bash
run_011_train_xvector.sh --config-file config_resnext50_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.01.sh
run_030_extract_xvectors.sh --config-file config_resnext50_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.01.sh
...
run_042a_eval_be_v3.sh --config-file config_resnext50_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.01.sh
...
```

## Recipe Steps

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
        all training and test data

   - `run_002a_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_002b_compute_fbank.sh`
      - Computes log-filter-banks for all datasets

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
      - Trains the x-vector network on English 4sec utts

   - `run_012_finetune_xvector.sh`
      - Fine-tune x-vector network on English 10-60 secs utts

   - `run_013_prepare_xvec_adapt_data.sh`
      - Prepares the Arabic adaptation data for x-vector training

   - `run_014_finetune_xvector_lastlayer_indomain.sh`
      - Finetunes last affine layer before x-vector embedding
        on Arabic data using deep-feat-prior regularization
      - It starts from model obtained in step 12

   - `run_015_finetune_xvector_full_indomain.sh`
      - Finetunes full x-vector network
        on Arabic data using deep-feat-prior regularization
      - It starts from model obtained in step 14

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for PLDA training and eval using English model from step 11

   - `run_031_extract_xvectors_ft1.sh`
      - Extracts x-vectors for PLDA training and eval using English finetuned model from step 12

   - `run_032_extract_xvectors_ft2.sh`
      - Extracts x-vectors for PLDA training and eval using Arabic finetuned model from step 14

   - `run_033_extract_xvectors_ft3.sh`
      - Extracts x-vectors for PLDA training and eval using Arabic finetuned model from step 15

   - `run_040a_eval_be_v1.sh, run_041a_eval_be_v2.sh,  run_042a_eval_be_v3.sh, run_043a_eval_be_v3_with_aug.sh`
      - Evals 3 different back-ends on the English x-vectors
           - V1: LDA + LNorm + PLDA adapted with SRE18 unlabeled
	   - V2: LDA + LNorm + PLDA adapted with SRE18 labeled
	   - V3: CORAL + LDA + LNorm + PLDA adapted with SRE18 labeled+unlabeled
	   - V3-Aug: CORAL + LDA + LNorm + PLDA adapted with SRE18 labeled+unlabeled + Noise Augmentation
      - Results are left in `exp/scores/exp/scores/resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1/*/plda_snorm300_cal_v1eval40/*_results`

   - `run_050a_eval_be_v1_ftxvec1.sh, run_051a_eval_be_v2_ftxvec1.sh,  run_052a_eval_be_v3_ftxvec1.sh, run_053a_eval_be_v3_with_aug_ftxvec1.sh`
      - Some as previous but on adapted x-vectors from step 12
      - Results are left in `exp/scores/resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.ft_1000_6000_sgdcos_lr0.05_b128_amp.v2/*/plda_snorm300_cal_v1eval40/*_results`

   - `run_060a_eval_be_v1_ftxvec2.sh, run_061a_eval_be_v2_ftxvec2.sh,  run_062a_eval_be_v3_ftxvec2.sh, run_063a_eval_be_v3_with_aug_ftxvec2.sh`
      - Some as previous but on adapted x-vectors from step 14
      - Results are left in `exp/scores/resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.ft_1000_6000_sgdcos_lr0.05_b128_amp.v2.ft_eaffine_rege_w0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2/*/plda_snorm300_cal_v1eval40/*_results`

   - `run_070a_eval_be_v1_ftxvec3.sh, run_071a_eval_be_v2_ftxvec3.sh,  run_072a_eval_be_v3_ftxvec3.sh, run_073a_eval_be_v3_with_aug_ftxvec3.sh`
      - Some as previous but on adapted x-vectors from step 15
      - Results are left in `exp/scores/resnet34_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.ft_1000_6000_sgdcos_lr0.05_b128_amp.v2.ft_eaffine_rege_w0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc0.1_we0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2/*/plda_snorm300_cal_v1eval40/*_results`


## Results

### Results using Back-end V3 with S-Norm

| Config | Model Type | Model Details | Fine-tuning | Back-end | SRE18 Eval 40% | | | SRE19 Progress | | | SRE19 Eval  | | |
| ------ | ---------- | ------------- | ------- | -------- | :------------: | :-: | :-: | :------------: | :-: | :-: | :------------: | :-: | :-: |
| |  |  |  | | EER(%) | MinDCF | ActDCF |  EER(%) | MinDCF | ActDCF |  EER(%) | MinDCF | ActDCF |
| config_resnet34_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.1.sh | ResNet34 | ArcFace s=30 / m=0.3 | N | V3(S-Norm) | 3.76 | 0.306 | 0.313 | 3.78 | 0.291 | 0.296 | 3.78 | 0.346 | 0.348 |
| | | | 1 | V3 (S-Norm) | 3.31 | 0.281 | 0.287 | 3.34 | 0.251 | 0.259 | 3.41 | 0.303 | 0.304 | 
| | | | 2 | V3 (S-Norm) | 3.18 | 0.282 | 0.290 | 3.18 | 0.241 | 0.256 | 3.32 | 0.299 | 0.300 |
| | | | 3 | V3 (S-Norm) | 2.92 | 0.240 | 0.253 | 2.88 | 0.225 | 0.232 | 2.92 | 0.264 | 0.267 |

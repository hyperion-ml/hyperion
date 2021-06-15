# VOiCES Challenge V1

x-Vector recipe for VOiCES Challenge
This setup is similar to the one used by JHU team for the actual challenge
This version uses Pytorch x-vectors and computes features on-the-fly from audio

## Citing

   This recipe is based on these works
```
@inproceedings{Snyder2019,
address = {Graz, Austria},
author = {Snyder, David and Villalba, Jes{\'{u}}s and Chen, Nanxin and Povey, Daniel and Sell, Gregory and Dehak, Najim and Khudanpur, Sanjeev},
booktitle = {Proceedings of the 20th Annual Conference of the International Speech Communication Association, INTERSPEECH 2019},
file = {:Users/villalba/Documents/Mendeley Desktop/Snyder et al. - 2019 - The JHU Speaker Recognition System for the VOiCES 2019 Challenge.pdf:pdf},
month = {sep},
publisher = {ISCA},
title = {{The JHU Speaker Recognition System for the VOiCES 2019 Challenge}},
year = {2019}
}
```

## Training Data
   - VOiCES protocol (default, follows the evaluation rules, cannot use MUSAN babble):
     - x-Vector network and PLDA back-end is trained on Voxceleb1+2 dev+test + SITW with augmentations
        - MUSAN noise, music and Mixer6 Babble (following eval protocol) or
        - RIR reverberation
   - SRE19-AV protocol (To use with networks trained on the SRE19-AV recipe)
     - x-Vector network and PLDA back-end is trained on Voxceleb1+2 dev+test with augmentations
        - MUSAN noise, music and babble
        - RIR reverberation

## Test data

   We evaluate:
     - VOiCES challenge

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
       - Mixer6
       - SITW for training
       - Voices dev+eval

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_003_prepare_noises_rirs.sh`
      - Prepares MUSAN noises, music to be used by SpeechAugment class.
      - Creates Babble noise from MUSAN and Mixer6 speech to be used by SpeechAugment class.
      - Prepares RIRs by compacting then into HDF5 files, to be used by SpeechAugment class.

   - `run_010_prepare_xvec_train_data.sh`
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_xvector.sh`
      - Trains the x-vector network

   - `run_030_extract_xvectors.sh`
      - Computes x-vectors for all datasets

   - `run_040_eval_be_v1_wo_diar.sh`
      - Trains back-end LDA+CenterWhiten+Lenght-Norm+PLDA on VoxCeleb
      - Evals back-end on all datasets without using diarization (assuming one speaker per test file) without AS-Norm
   - `run_040_eval_backend.sh`
      - Trains back-end LDA+CenterWhiten+Lenght-Norm+PLDA on VoxCeleb(+SITW)
      - It uses Voices dev for centering
      - Evals Voices dev and eval with and without S-Norm
      - Calibrates on voices dev
   - `run_041_eval_backend_2folds.sh`
      - Evaluate the dev without cheating centering and calibration splitting into 2 folds
      - Fold 1 is centered and calibrated with fold2 and viceversa
      - Eval is centered and calibrated with both folds
   - `run_042_eval_backend_adapt_2folds.sh`
      - Same as 41, backend withinn class cov is adapted to voxceleb dev
   - `run_050_make_res_tables.sh`
      - Prints table with results from steps 40-41
      - Usage: 
         `run_050_make_res_tables.sh exp/scores/fbank80_stmn_resnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1/lda200_splday150_v1_voxcelebcat_augx6`
   - `run_051_make_adapt_res_tables.sh`
      - Prints table with results from step 42
      - Usage: 
         `run_051_make_adapt_res_tables.sh exp/scores/fbank80_stmn_resnet34_e256_arcs30m0.3_do0_adam_lr0.05_b512_amp.v1/lda200_splday150_v2_voxcelebcat_augx6_adapt_w0.35`

## Results

Results here are from step `run_041...`

| Config | NNet | AS-Norm | VOICES DEV |    |        | VOICES EVAL |      |      |
| ------ | ---- | :-----: | :--------: | :--: | :--: | :---------: | :--: | :--: | 
|  |  |  | EER  | MinDCF  | ActDCF | EER | MinDCF | ActDCF | 
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp_sre19prot.v1.sh |  ResNet34 | N | 1.09 | 0.109 | 0.110 | 4.12 | 0.281 | 0.288 | 
| config_fbank80_stmn_res2net50w26s4_arcs30m0.3_adam_lr0.05_amp_sre19prot.v1.sh | Res2Net50 | N | 1.04 | 0.089 | 0.089 | 3.50 | 0.249 | 0.250 | 

# VoxCeleb V2

Recipe for the VoxCeleb Speaker Verification Task using Wav2Vec2, WavLM or Hubert models from HuggingFace as feature extractors

## Differences w.r.t VoxCeleb V1 recipe

## Citing

## Training Data

   - x-Vector network is trained on Voxceleb2 dev + test with augmentations
     - MUSAN noise
     - RIR reverberation

## Test data

   - Test data is VoxCeleb 1
   - We evaluate 6 conditions:
      - VoxCeleb-O (Original): Original Voxceleb test set with 40 speakers
      - Voxceleb-O-cleaned: VoxCeleb-O cleaned-up of some errors
      - VoxCeleb-E (Entire): List using all utterances of VoxCeleb1
      - Voxceleb-E-cleaned: VoxCeleb-E cleaned-up of some errors
      - VoxCeleb-H (Hard): List of hard trials between all utterances of VoxCeleb1, same gender and nationality trials.
      - Voxceleb-H-cleaned: VoxCeleb-H cleaned-up of some errors

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it will use 
   - For better performance use 
```bash
run_011_train_xvector.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_030_extract_xvectors.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh --use-gpu true
run_040_eval_be.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
```


## Recipe Steps:

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
          - VoxCeleb2 train+test
          - VoxCeleb1 O/E/H eval sets

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
      - Trains the x-vector model on frozen wav2vec features
      - Finetunes wav2vec+x-vector model
      - Large margin finetuning of wav2vec+x-vector model

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Exctracts x-vectors for VoxCeleb1 test sets

   - `run_040_eval_be.sh`
      - Trains PLDA and evals PLDA and cosine scoring back-ends


## Results

### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.84 | 0.060 | 0.116 |
| | | | Cosine + AS-Norm | 0.81 | 0.058 | 0.108 |
| | | | Cosine + QMF | 0.75 | 0.054 | 0.086 |

### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.81 | 0.051 | 0.087 |
| | | | Cosine + AS-Norm | 0.78 | 0.047 | 0.083 |
| | | | Cosine + QMF | 0.75 | 0.046 | 0.076 |

### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.73 | 0.113 | 0.182 |
| | | | Cosine + AS-Norm | 1.63 | 0.100 | 0.160 |
| | | | Cosine + QMF | 1.56 | 0.096 | 0.155 |

### VoxSRC2022 dev

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.60 | 0.163 | 0.257 |
| | | | Cosine + AS-Norm | 2.43 | 0.150 | 0.244 |
| | | | Cosine + QMF | 2.31 | 0.143 | 0.232 |

# VoxCeleb V1.2

Recipe for the VoxCeleb Speaker Verification Task

## Differences w.r.t VoxCeleb V1 recipe

In recipe version V1: 
   - We compute speech augmentations and acoustic features offline and dump them to disk. 
   - Augmentation is performed using Kaldi scripts and wav-reverbate tool
   - Babble noise is created on-the-fly when computing features by mixing 3-7 single speaker files.

In V1.1:
   - We compute speech augmentations and acoustic features are computed always on-the-fly,
     we don't dump any features to disk. 
   - Augmentation is performed using Hyperin SpeechAugment class.
   - The behavior of this class is controlled 
     by the the configuration file `conf/reverb_noise_aug.yml`, 
     which mimics the proportions of noise and RIR types, and SNRs used in the V1 or the recipe.
   - Babble noise is created offline by mixing 3-10 single speaker files.

In V1.2:
   - Feaure extractor is embedded into the pytorch model in classes derived from Wav2XVector base class.
   - Kaldi format is replaced by new format based on pandas tables
   - Kaldi style bash scripts are removed and replaced by python scripts
   - Most python scripts are called using Hyperion entry points 

## Citing

## Training Data

   - x-Vector network is trained on Voxceleb2 dev + test with augmentations
     - MUSAN noise
     - RIR reverberation

## Test data

   - Test data is VoxCeleb 1
   - We evaluate the 3 conditions (with cleaned lists):
      - VoxCeleb-O (Original): Original Voxceleb test set with 40 speakers
      - VoxCeleb-E (Entire): List using all utterances of VoxCeleb1
      - VoxCeleb-H (Hard): List of hard trials between all utterances of VoxCeleb1, same gender and nationality trials.
 

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it will use Light ResNet (16 base channels)
   - For better performance use full ResNet (64 base channels) using `config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh` file as
```bash
run_005_train_xvector.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_006_extract_xvectors.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh --use-gpu true
run_007_eval_be.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
```

   - To train with mixed precision training use config file `config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh`

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

   - `run_004_prepare_xvec_train_data.sh`
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_005_train_xvector.sh`
      - Trains the x-vector network

   - `run_006_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Exctracts x-vectors for VoxCeleb1 test sets

   - `run_007_eval_be.sh`
      - Trains PLDA and evals PLDA and cosine scoring back-ends

## Results

### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_ecapatdnn512x3.v3.0.sh | ECAPA-TDNN 512x3 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.11 | 0.069 | 0.126 |
| | | | Cosine + AS-Norm | 1.10 | 0.065 | 0.108 |
| | | | Cosine + QMF | 0.95 | 0.059 | 0.084 |
| config_fbank80_stmn_ecapatdnn512x3.v3.1.sh | ECAPA-TDNN 512x3 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.02 | 0.067 | 0.109 |
| | | | Cosine + AS-Norm | 0.98 | 0.062 | 0.092 |
| | | | Cosine + QMF | 0.85 | 0.061 | 0.091 | 
| config_fbank80_stmn_ecapatdnn2048x4.v3.0.sh | ECAPA-TDNN 2048x4 | Stage2: ArcFace m=0.3/intertop_m=0.1 Dropout=0.25 | Cosine | 0.68 | 0.052 | 0.088 |
| | | | Cosine + AS-Norm | 0.63 | 0.049 | 0.083 |
| | | | Cosine + QMF | 0.57 | 0.037 | 0.071 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.1.sh | ECAPA-TDNN 2048x4 | Stage2: Subcenter ArcFace m=0.3/intertop_m=0.1/centers=2 Dropout=0.25 | Cosine | 0.62 | 0.049 | 0.076 |
| | | | Cosine + AS-Norm | 0.61 | 0.044 | 0.075 |
| | | | Cosine + QMF | 0.53 | 0.037 | 0.076 |
| config_fbank80_stmn_lresnet34.v3.1.sh | Thin-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.59 | 0.1 | 0.172 |
| | | | Cosine + AS-Norm | 1.54 | 0.927 | 0.140 |
| | | | Cosine + QMF | 1.32 | 0.083 | 0.121 |
| config_fbank80_stmn_resnet34.v3.0.sh | ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.77 | 0.048 | 0.071 |
| | | | Cosine + AS-Norm | 0.70 | 0.039 | 0.048 |
| | | | Cosine + QMF | 0.62 | 0.034 | 0.042 |
| config_fbank80_stmn_resnet34.v3.1.sh | ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.68 | 0.039 | 0.048
| | | | Cosine + AS-Norm | 0.60 | 0.036 | 0.052 |
| | | | Cosine + QMF | 0.53 | 0.033 | 0.050 |
| config_fbank80_stmn_cwseresnet34.v3.0.sh | CwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.76 | 0.048 | 0.071 |
| | | | Cosine + AS-Norm | 0.70 | 0.041 | 0.061 |
| | | | Cosine + QMF | 0.62 | 0.037 | 0.056 |
| config_fbank80_stmn_fwseresnet34.v3.0.sh | FwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.77 | 0.48 | 0.077 |
| | | | Cosine + AS-Norm | 0.68 | 0.040 | 0.062|
| | | | Cosine + QMF | 0.62 | 0.036 | 0.063 |
| config_fbank80_stmn_fwseresnet34.v3.1.sh | FwSE-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.66 | 0.046 | 0.060 |
| | | | Cosine + AS-Norm | 0.61 | 0.040 | 0.052 |
| | | | Cosine + QMF | 0.057 | 0.037 | 0.058 |
| config_fbank80_stmn_fwseresnet34pe.v3.1.sh | FwSE-ResNet34-FPE | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.73 | 0.042 | 0.053 |
| | | | Cosine + AS-Norm | 0.64 | 0.034 | 0.047 |
| | | | Cosine + QMF | 0.60 | 0.033 | 0.044 |
| config_fbank80_stmn_tseresnet34.v3.0.sh | Time-SE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.78 | 0.053 | 0.082 |
| | | | Cosine + AS-Norm | 0.70 | 0.043 | 0.076 |
| | | | Cosine + QMF | 0.63 | 0.042 | 0.071 |
| config_fbank80_stmn_cfwseresnet34.v3.0.sh | CwFwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.78 | 0.051 | 0.095 |
| | | | Cosine + AS-Norm | 0.72 | 0.046 | 0.070 |
| | | | Cosine + QMF | 0.67 | 0.039 | 0.074 |
| config_fbank80_stmn_idrnd_resnet100.v3.0.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.56 | 0.040 | 0.065 |
| | | | Cosine + AS-Norm | 0.52 | 0.033 | 0.045 |
| | | | Cosine + QMF | 0.45 | 0.027 | 0.043 |
| config_fbank80_stmn_idrnd_resnet100.v3.1.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.50 | 0.035 | 0.038 |
| | | | Cosine + AS-Norm | 0.47 | 0.031 | 0.038 |
| | | | Cosine + QMF | 0.40 | 0.027 | 0.032 |
| config_fbank80_stmn_idrnd_resnet100.v3.2.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.49 | 0.032 | 0.038 |
| | | | Cosine + AS-Norm | 0.43 | 0.025 | 0.034 |
| | | | Cosine + QMF | 0.37 | 0.024 | 0.033 |
| config_fbank80_stmn_res2net50w26s8.v3.0.sh | Res2Net50 w26 scale=8 | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.60 | 0.043 | 0.071 |
| | | | Cosine + AS-Norm | 0.53 | 0.034 | 0.063 |
| | | | Cosine + QMF | 0.49 | 0.033 | 0.054 |

### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_ecapatdnn512x3.v3.0.sh | ECAPA-TDNN 512x3 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.16 | 0.073 | 0.130 |
| | | | Cosine + AS-Norm | 1.13 | 0.068 | 0.118 |
| | | | Cosine + QMF | 1.06 | 0.064 | 0.112 |
| config_fbank80_stmn_ecapatdnn512x3.v3.1.sh | ECAPA-TDNN 512x3 | Stage2: SubCenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.06 | 0.066 | 0.116 |
| | | | Cosine + AS-Norm | 1.01 | 0.061 | 0.106 |
| | | | Cosine + QMF | 0.96 | 0.058 | 0.097 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.0.sh | ECAPA-TDNN 2048x4 | Stage2: ArcFace m=0.3/intertop_m=0.1 Dropout=0.25 | Cosine | 0.85 | 0.055 | 0.100 |
| | | | Cosine + AS-Norm | 0.80 | 0.050 | 0.087 |
| | | | Cosine + QMF | 0.76 | 0.047 | 0.083 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.1.sh | ECAPA-TDNN 2048x4 | Stage2: Subcenter ArcFace m=0.3/intertop_m=0.1/centers=2 Dropout=0.25 | Cosine | 0.83 | 0.052 | 0.096 |
| | | | Cosine + AS-Norm | 0.77 | 0.049 | 0.086 |
| | | | Cosine + QMF | 0.74 | 0.047 | 0.082 |
| config_fbank80_stmn_lresnet34.v3.1.sh | Thin-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.69 | 0.103 | 0.174 |
| | | | Cosine + AS-Norm | 1.62 | 0.096 | 0.156 |
| | | | Cosine + QMF | 1.51 | 0.091 | 0.152 |
| config_fbank80_stmn_resnet34.v3.0.sh | ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.86 | 0.054 | 0.098 |
| | | | Cosine + AS-Norm | 0.81 | 0.049 | 0.087 |
| | | | Cosine + QMF | 0.77 | 0.046 | 0.082  |
| config_fbank80_stmn_resnet34.v3.1.sh | ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.80 | 0.049 | 0.094 |
| | | | Cosine + AS-Norm | 0.76 | 0.046 | 0.081 |
| | | | Cosine + QMF | 0.70 | 0.043 | 0.074 |
| config_fbank80_stmn_cwseresnet34.v3.0.sh | CwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.89 | 0.058 | 0.098 |
| | | | Cosine + AS-Norm | 0.84 | 0.053 | 0.087|
| | | | Cosine + QMF | 0.80 | 0.050  | 0.081 |
| config_fbank80_stmn_fwseresnet34.v3.0.sh | FwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.83 | 0.053 | 0.098 |
| | | | Cosine + AS-Norm | 0.78 | 0.047| 0.085 |
| | | | Cosine + QMF | 0.74 | 0.045 | 0.081 |
| config_fbank80_stmn_fwseresnet34.v3.1.sh | FwSE-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.80 | 0.051 | 0.090 |
| | | | Cosine + AS-Norm | 0.74 | 0.046 | 0.081 |
| | | | Cosine + QMF | 0.70 | 0.044 | 0.076 |
| config_fbank80_stmn_fwseresnet34pe.v3.1.sh | FwSE-ResNet34-FPE | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.80 | 0.052 | 0.094 |
| | | | Cosine + AS-Norm | 0.76 | 0.047 | 0.081 |
| | | | Cosine + QMF | 0.72 | 0.045 | 0.076 |
| config_fbank80_stmn_tseresnet34.v3.0.sh | Time-SE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.91 | 0.057 | 0.100 |
| | | | Cosine + AS-Norm | 0.85 | 0.052 | 0.089 |
| | | | Cosine + QMF | 0.81 | 0.049 | 0.085 |
| config_fbank80_stmn_cfwseresnet34.v3.0.sh | CwFwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 0.94 | 0.059 | 0.105 |
| | | | Cosine + AS-Norm | 0.88 | 0.053 | 0.093 |
| | | | Cosine + QMF | 0.84 | 0.051 | 0.088 |
| config_fbank80_stmn_idrnd_resnet100.v3.0.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.71 | 0.044 | 0.076|
| | | | Cosine + AS-Norm | 0.66 | 0.040 | 0.069 |
| | | | Cosine + QMF | 0.63 | 0.037 | 0.067 |
| config_fbank80_stmn_idrnd_resnet100.v3.1.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.69 | 0.043 | 0.074 |
| | | | Cosine + AS-Norm | 0.65 | 0.039 | 0.068 |
| | | | Cosine + QMF | 0.63 | 0.036 | 0.065 |
| config_fbank80_stmn_idrnd_resnet100.v3.2.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 0.66 | 0.040 | 0.072 |
| | | | Cosine + AS-Norm | 0.62 | 0.037 | 0.066 |
| | | | Cosine + QMF | 0.59 | 0.035 | 0.064 |
| config_fbank80_stmn_res2net50w26s8.v3.0.sh | Res2Net50 w26 scale=8 | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.75 | 0.047 | 0.077 |
| | | | Cosine + AS-Norm | 0.70 | 0.042 | 0.072 |
| | | | Cosine + QMF | 0.68 | 0.040 | 0.069 |


### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_ecapatdnn512x3.v3.0.sh | ECAPA-TDNN 512x3 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.10 | 0.128 | 0.209 |
| | | | Cosine + AS-Norm | 1.99 | 0.118 | 0.190 |
| | | | Cosine + QMF | 1.84 | 0.111 | 0.184 |
| config_fbank80_stmn_ecapatdnn512x3.v3.1.sh | ECAPA-TDNN 512x3 | Stage2: SubCenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.93 | 0.120 | 0.198 |
| | | | Cosine + AS-Norm | 1.84 | 0.113 | 0.184 |
| | | | Cosine + QMF | 1.73 | 0.108 | 0.177 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.0.sh | ECAPA-TDNN 2048x4 | Stage2: ArcFace m=0.3/intertop_m=0.1 Dropout=0.25 | Cosine | 1.66 | 0.103 | 0.168 |
| | | | Cosine + AS-Norm | 1.53 | 0.091 | 0.151 |
| | | | Cosine + QMF | 1.44 | 0.087 | 0.145 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.1.sh | ECAPA-TDNN 2048x4 | Stage2: Subcenter ArcFace m=0.3/intertop_m=0.1/centers=2 Dropout=0.25 | Cosine | 1.65 | 0.0101 | 0.169 |
| | | | Cosine + AS-Norm | 1.53 | 0.090 | 0.149 |
| | | | Cosine + QMF | 1.46 | 0.087 | 0.144 |
| config_fbank80_stmn_lresnet34.v3.1.sh | Thin-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.84 | 0.167 | 0.267 |
| | | | Cosine + AS-Norm | 2.58 | 0.150 | 0.252 |
| | | | Cosine + QMF | 2.45 | 0.144 | 0.234 |
| config_fbank80_stmn_resnet34.v3.0.sh | ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.62 | 0.098 | 0.164 |
| | | | Cosine + AS-Norm | 1.45 | 0.085 | 0.142 |
| | | | Cosine + QMF | 1.36 | 0.082 | 0.137 |
| config_fbank80_stmn_resnet34.v3.1.sh | ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.56 | 0.091 | 0.157 |
| | | | Cosine + AS-Norm | 1.40 | 0.080 | 0.135 |
| | | | Cosine + QMF | 1.33 | 0.076 | 0.128 |
| config_fbank80_stmn_cwseresnet34.v3.0.sh | CwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.70 | 0.1 |  0.165 |
| | | | Cosine + AS-Norm | 1.50 | 0.086 | 0.138 |
| | | | Cosine + QMF | 1.44 | 0.085  | 0.139 |
| config_fbank80_stmn_fwseresnet34.v3.0.sh | FwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.59 | 0.096 | 0.165 |
| | | | Cosine + AS-Norm | 1.41 | 0.083 | 0.143 |
| | | | Cosine + QMF | 1.34 | 0.079 | 0.136 |
| config_fbank80_stmn_fwseresnet34.v3.1.sh | FwSE-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.58 | 0.096 | 0.162 |
| | | | Cosine + AS-Norm | 1.43 | 0.083 | 0.140 |
| | | | Cosine + QMF | 0.134 | 0.079 | 0.134 |
| config_fbank80_stmn_fwseresnet34pe.v3.1.sh | FwSE-ResNet34-FPE | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.61 | 0.097 | 1.63 |
| | | | Cosine + AS-Norm | 1.44 | 0.085 | 0.138 |
| | | | Cosine + QMF | 1.37 | 0.080 | 0.132 |
| config_fbank80_stmn_tseresnet34.v3.0.sh | Time-SE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.75 | 0.104 | 0.171 |
| | | | Cosine + AS-Norm | 1.56 | 0.091 | 0.152 |
| | | | Cosine + QMF | 1.50 | 0.087 | 0.145 |
| config_fbank80_stmn_cfwseresnet34.v3.0.sh | CwFwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 1.76 | 0.104 | 0.174 |
| | | | Cosine + AS-Norm |  1.58 | 0.092 | 0.152 |
| | | | Cosine + QMF | 1.51 | 0.089 | 0.149 |
| config_fbank80_stmn_idrnd_resnet100.v3.0.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.30 | 0.076 | 0.125 |
| | | | Cosine + AS-Norm | 1.15 | 0.066 | 0.109 |
| | | | Cosine + QMF | 1.11 | 0.065 | 0.103 |
| config_fbank80_stmn_idrnd_resnet100.v3.1.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.36 | 0.077 | 0.122 |
| | | | Cosine + AS-Norm | 1.23 | 0.069 | 0.112 |
| | | | Cosine + QMF | 1.17 | 0.065 | 0.110 |
| config_fbank80_stmn_idrnd_resnet100.v3.1.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.27 | 0.072 | 0.121 |
| | | | Cosine + AS-Norm | 1.15 | 0.065 | 0.107 |
| | | | Cosine + QMF | 1.10 | 0.062 | 0.102 |
| config_fbank80_stmn_res2net50w26s8.v3.0.sh | Res2Net50 w26 scale=8 | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.41 | 0.081 | 0.132 |
| | | | Cosine + AS-Norm | 1.28 | 0.071 | 0.116 |
| | | | Cosine + QMF | 1.21 | 0.069 | 0.113 |


### VoxSRC2022 dev

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_ecapatdnn512x3.v3.0.sh | ECAPA-TDNN 512x3 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.87 | 0.185 | 0.304 |
| | | | Cosine + AS-Norm | 2.84 | 0.182 | 0.304 |
| | | | Cosine + QMF | 2.61 | 0.172 | 0.283 |
| config_fbank80_stmn_ecapatdnn512x3.v3.1.sh | ECAPA-TDNN 512x3 | Stage2: SubCenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.60 | 0.174 | 0.287 |
| | | | Cosine + AS-Norm | 2.58 | 0.172 | 0.291 | 
| | | | Cosine + QMF | 2.44 | 0.161 | 0.274 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.0.sh | ECAPA-TDNN 2048x4 | Stage2: ArcFace m=0.3/intertop_m=0.1 Dropout=0.25 | Cosine | 2.33 | 0.156 | 0.260 |
| | | | Cosine + AS-Norm | 2.19 | 0.144 | 0.263 |
| | | | Cosine + QMF | 2.06 | 0.137 | 0.251 |
| config_fbank80_stmn_ecapatdnn2048x4.v3.1.sh | ECAPA-TDNN 2048x4 | Stage2: Subcenter ArcFace m=0.3/intertop_m=0.1/centers=2 Dropout=0.25 | Cosine | 2.34 | 0.152 | 0.275 |
| | | | Cosine + AS-Norm | 2.24 | 0.143 | 0.268 |
| | | | Cosine + QMF | 2.12 | 0.139 | 0.255 |
| config_fbank80_stmn_lresnet34.v3.1.sh | Thin-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 3.74 | 0.239 | 0.394 |
| | | | Cosine + AS-Norm | 3.45 | 0.225 | 0.377 |
| | | | Cosine + QMF | 3.27 | 0.213 | 0.356 |
| config_fbank80_stmn_resnet34.v3.0.sh | ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.19 | 0.142 | 0.242 |
| | | | Cosine + AS-Norm | 2.00 | 0.133 | 0.254 |
| | | | Cosine + QMF | 1.86 | 0.126 | 0.229 |
| config_fbank80_stmn_resnet34.v3.1.sh | ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.15 | 0.135 | 0.233 |
| | | | Cosine + AS-Norm | 1.98 | 0.126 | 0.245 |
| | | | Cosine + QMF | 1.86 | 0.119 | 0.222 |
| config_fbank80_stmn_cwseresnet34.v3.0.sh | CwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.34 | 0.145 | 0.246 |
| | | | Cosine + AS-Norm | 2.10 | 0.135 | 0.248 |
| | | | Cosine + QMF | 2.01 | 0.127 | 0.218 |
| config_fbank80_stmn_fwseresnet34.v3.0.sh | FwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.25 | 0.136 | 0.239 |
| | | | Cosine + AS-Norm | 1.99 | 0.127 | 0.232 |
| | | | Cosine + QMF | 1.87 | 0.119 | 0.216 |
| config_fbank80_stmn_fwseresnet34.v3.1.sh | FwSE-ResNet34 | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.14 | 0.134 | 0.228 |
| | | | Cosine + AS-Norm | 1.97 | 0.124 | 0.223 |
| | | | Cosine + QMF | 1.82 | 0.116 | 0.205 |
| config_fbank80_stmn_fwseresnet34pe.v3.1.sh | FwSE-ResNet34-FPE | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.27 | 0.138 | 0.238 |
| | | | Cosine + AS-Norm | 2.08 | 0.129 | 0.223 |
| | | | Cosine + QMF | 1.94 | 0.120 | 0.207 |
| config_fbank80_stmn_tseresnet34.v3.0.sh | Time-SE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.36 | 0.153 | 0.259 |
| | | | Cosine + AS-Norm | 2.18 | 0.139 | 0.249 |
| | | | Cosine + QMF | 2.08 | 0.128 | 0.222 |
| config_fbank80_stmn_cfwseresnet34.v3.0.sh | CwFwSE-ResNet34 | Stage2: ArcFace m=0.3/intertop_m=0.1 | Cosine | 2.49 | 0.158 | 0.265 |
| | | | Cosine + AS-Norm | 2.29 | 0.145 | 0.251 |
| | | | Cosine + QMF | 2.17 | 0.133 | 0.230 |
| config_fbank80_stmn_idrnd_resnet100.v3.0.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.92 | 0.124 | 0.208 |
| | | | Cosine + AS-Norm | 1.71 | 0.109 | 0.212 |
| | | | Cosine + QMF | 1.62 | 0.103 | 0.192 |
| config_fbank80_stmn_idrnd_resnet100.v3.1.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 2.02 | 0.116 | 0.194 |
| | | | Cosine + AS-Norm | 1.81 | 0.107 | 0.199 |
| | | | Cosine + QMF | 1.72 | 0.099 | 0.186 |
| config_fbank80_stmn_idrnd_resnet100.v3.2.sh | ResNet100 / BasicBlock 128-256 ch. | Stage2: Subcenter-ArcFace m=0.3/intertop_m=0.1/centers=2 | Cosine | 1.91 | 0.111 | 0.192 |
| | | | Cosine + AS-Norm | 1.75 | 0.105 | 0.194 |
| | | | Cosine + QMF | 1.64 | 0.098 | 0.181 |
| config_fbank80_stmn_res2net50w26s8.v3.0.sh | Res2Net50 w26 scale=8 | Stage2: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.96 | 0.124 | 0.211 |
| | | | Cosine + AS-Norm | 1.79 | 0.118 | 0239 |
| | | | Cosine + QMF | 1.68 | 0.114 | 0.216 |

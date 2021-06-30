# VoxCeleb V1.1

Recipe for the VoxCeleb Speaker Verification Task

## Differences w.r.t VoxCeleb V1 recipe

In recipe version V1: 
   - We compute speech augmentations and acoustic features offline and dump them to disk. 
   - Augmentation is performed using Kaldi scripts and wav-reverbate tool
   - Babble noise is created on-the-fly when computing features by mixing 3-7 single speaker files.

In this recipe:
   - We compute speech augmentations and acoustic features are computed always on-the-fly,
     we don't dump any features to disk. 
   - Augmentation is performed using Hyperin SpeechAugment class.
   - The behavior of this class is controlled 
     by the the configuration file `conf/reverb_noise_aug.yml`, 
     which mimics the proportions of noise and RIR types, and SNRs used in the V1 or the recipe.
   - Babble noise is created offline by mixing 3-10 single speaker files.


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
   - By default it will use Light ResNet (16 base channels)
   - For better performance use full ResNet (64 base channels) using `config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh` file as
```bash
run_011_train_xvector.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_030_extract_xvectors.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh --use-gpu true
run_040_eval_be.sh --config-file config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
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

   - `run_010_prepare_xvec_train_data.sh`
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_xvector.sh`
      - Trains the x-vector network

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Exctracts x-vectors for VoxCeleb1 test sets

   - `run_040_eval_be.sh`
      - Trains PLDA and evals PLDA and cosine scoring back-ends


## Results

### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.00 | 0.129 | 0.216 |
| | | | Cosine | 2.04 | 0.138 | 0.210 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA |  1.35 | 0.091 | 0.159 |
| | | | Cosine |  1.22 | 0.082 | 0.129 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp_swa.v1.sh | ResNet34 | + SWA | Cosine | 1.19 | 0.074 | 0.124 |
| config_fbank80_stmn_resnet50_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet50 | ArcFace s=30/m=0.3 | PLDA |  1.30 | 0.090 | 0.160 |
| | | | Cosine |  1.44 | 0.100 | 0.173 |
| config_fbank80_stmn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.23 | 0.091 | 0.143 |
| | | | Cosine |  1.17 | 0.081 | 0.110 |
| config_fbank80_stmn_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 v2 | EfficientNet-b4 with strides=1122121 <br> ArcFace s=30/m=0.3 | 1.37 | 0.104 | 0.179 |
| | | | Cosine | 1.31 | 0.080 | 0.139 |
| config_fbank80_stmn_effnetb7_v2_eina_hln_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b7 v2 | EfficientNet-b7 with strides=1122121 <br> Instance-Norm with affine transform in Encoder <br> Layer-Norm in head <br> ArcFace s=30/m=0.3 | 1.29 | 0.088 | 0.129 |
| | | | Cosine | 1.23 | 0.083 | 0.136 |
| config_fbank80_stmn_res2net34w16s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=16x4 | ArcFace s=30/m=0.3 | PLDA |  1.20 | 0.095 | 0.156 |
| | | | Cosine | 1.29 | 0.089 | 0.146 |
| config_fbank80_stmn_res2net34w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  1.20 | 0.084 | 0.136 |
| | | | Cosine | 1.18 | 0.078 | 0.115 |
| config_fbank80_stmn_res2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  1.11 | 0.084 | 0.145 |
| | | | Cosine | 1.12 | 0.073 | 0.131 |
| config_fbank80_stmn_seres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-Res2Net50 | se-r=16 <br> ArcFace s=30/m=0.3 | PLDA |  1.53 | 0.104 | 0.189 |
| | | | Cosine | 1.31 | 0.084 | 0.132 |
| config_fbank80_stmn_tseres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-Res2Net50 | se-r=256 <br> ArcFace s=30/m=0.3 | PLDA |  0.98 | 0.066 | 0.116 |
| | | | Cosine | 1.12 | 0.071 | 0.103 |
| config_fbank80_stmn_res2net50w13s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=13x8 | ArcFace s=30/m=0.3 | PLDA |  1.05 | 0.077 | 0.123 |
| | | | Cosine | 0.96 | 0.065 | 0.110 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x8 | ArcFace s=30/m=0.3 | PLDA |  1.04 | 0.071 | 0.118 |
| | | | Cosine | 0.93 | 0.067 | 0.108 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1_swa.sh | Res2Net50 width=26x8 | + SWA | PLDA |  0.90 | 0.067 | 0.118 |
| | | | Cosine | 0.85 | 0.060 | 0.094 |
| config_fbank80_stmn_spinenet49s_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49S | ArcFace s=30/m=0.3 | PLDA | 1.44 | 0.102 | 0.169 |
| | | | Cosine | 1.29 | 0.084 | 0.140 |
| config_fbank80_stmn_spinenet49_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49 | ArcFace s=30/m=0.3 | Cosine | 1.12 | 0.071 | 0.116 |
| config_fbank80_stmn_spine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 1.05 | 0.074 | 0.116 |
| config_fbank80_stmn_tsespine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 1.09 | 0.081 | 0.150 |


### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.86 | 0.124 | 0.210 |
| | | | Cosine | 1.93 | 0.122 | 0.201 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.43 | 0.091 | 0.159 |
| | | | Cosine | 1.24 | 0.080 | 0.136 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp_swa.v1.sh | ResNet34 | + SWA | Cosine | 1.19 | 0.077 | 0.132 |
| config_fbank80_stmn_resnet50_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet50 | ArcFace s=30/m=0.3 | PLDA | 1.27 | 0.084 | 0.150 |
| | | | Cosine | 1.30 | 0.082 | 0.150 |
| config_fbank80_stmn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.30 | 0.083 | 0.146 |
| | | | Cosine | 1.09 | 0.071 | 0.124 |
| config_fbank80_stmn_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 v2 | EfficientNet-b4 with strides=1122121 <br> ArcFace s=30/m=0.3 | 1.45 | 0.097 | 0.165 |
| | | | Cosine | 1.15 | 0.076 | 0.132 |
| config_fbank80_stmn_effnetb7_v2_eina_hln_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b7 v2 | EfficientNet-b7 with strides=1122121 <br> Instance-Norm with affine transform in Encoder <br> Layer-Norm in head <br> ArcFace s=30/m=0.3 | 1.47 | 0.094 | 0.165 |
| | | | Cosine | 1.27 | 0.082 | 0.148 |
| config_fbank80_stmn_res2net34w16s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=16x4 | ArcFace s=30/m=0.3 | PLDA |  1.31 | 0.086 | 0.149 |
| | | | Cosine | 1.22 | 0.079 | 0.134 |
| config_fbank80_stmn_res2net34w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  1.27 | 0.082 | 0.145 |
| | | | Cosine | 1.16 | 0.074 | 0.130 |
| config_fbank80_stmn_res2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  1.23 | 0.077 | 0.136 |
| | | | Cosine | 1.11 | 0.071 | 0.125 |
| config_fbank80_stmn_seres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-Res2Net50 | se-r=16 <br> ArcFace s=30/m=0.3 | PLDA |  1.46 | 0.097 | 0.173 |
| | | | Cosine | 1.24 | 0.080 | 0.140 |
| config_fbank80_stmn_tseres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-Res2Net50 | se-r=256 <br> ArcFace s=30/m=0.3 | PLDA |  1.11 | 0.071 | 0.127 |
| | | | Cosine | 1.05 | 0.067 | 0.117 |
| config_fbank80_stmn_res2net50w13s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=13x8 | ArcFace s=30/m=0.3 | PLDA |  1.23 | 0.078 | 0.134 |
| | | | Cosine | 1.05 | 0.069 | 0.121 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x8 | ArcFace s=30/m=0.3 | PLDA |  1.18 | 0.075 | 0.131 |
| | | | Cosine | 0.98 | 0.063 | 0.110 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp_swa.v1.sh | Res2Net50 width=26x8 | + SWA | PLDA |  1.17 | 0.072 | 0.123 |
| | | | Cosine | 0.94 | 0.061 | 0.107 |
| config_fbank80_stmn_spinenet49s_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49S | ArcFace s=30/m=0.3 | PLDA | 1.56 | 0.095 | 0.166 |
| | | | Cosine | 1.27 | 0.079 | 0.142 |
| config_fbank80_stmn_spinenet49_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49 | ArcFace s=30/m=0.3 | Cosine | 1.19 | 0.077 | 0.137 |
| config_fbank80_stmn_spine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 1.12 | 0.073 | 0.129 |
| config_fbank80_stmn_tsespine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | TSE-Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 1.05 | 0.068 | 0.120 |


### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 3.29 | 0.195 | 0.318 |
| | | | Cosine | 3.27 | 0.188 | 0.303 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.66 | 0.160 | 0.258 |
| | | | Cosine | 2.32 | 0.139 | 0.232 |
| config_fbank80_stmn_resnet34_arcs30m0.3_adam_lr0.05_amp_swa.v1.sh | ResNet34 | + SWA | Cosine | 2.19 | 0.133 | 0.215 |
| config_fbank80_stmn_resnet50_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet50 | ArcFace s=30/m=0.3 | PLDA | 2.33 | 0.139 | 0.227 |
| | | | Cosine | 2.33 | 0.142 | 0.235 |
| config_fbank80_stmn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.46 | 0.142 | 0.237 |
| | | | Cosine | 2.14 | 0.126 | 0.203 |
| config_fbank80_stmn_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 v2 | EfficientNet-b4 with strides=1122121 <br> ArcFace s=30/m=0.3 | 2.57 | 0.153 | 0.255 |
| | | | Cosine | 2.11 | 0.127 | 0.205 |
| config_fbank80_stmn_effnetb7_v2_eina_hln_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b7 v2 | EfficientNet-b7 with strides=1122121 <br> Instance-Norm with affine transform in Encoder <br> Layer-Norm in head <br> ArcFace s=30/m=0.3 | 2.64 | 0.157 | 0.244 |
| | | | Cosine | 2.33 | 0.141 | 0.232 |
| config_fbank80_stmn_res2net34w16s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=16x4 | ArcFace s=30/m=0.3 | PLDA |  2.42 | 0.144 | 0.245 |
| | | | Cosine | 2.26 | 0.133 | 0.224
| config_fbank80_stmn_res2net34w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net34 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  2.39 | 0.141 | 0.235 |
| | | | Cosine | 2.17 | 0.128 | 0.215
| config_fbank80_stmn_res2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x4 | ArcFace s=30/m=0.3 | PLDA |  2.28 | 0.131 | 0.225 |
| | | | Cosine | 2.11 | 0.124 | 0.204 |
| config_fbank80_stmn_seres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-Res2Net50 | se-r=16 <br> ArcFace s=30/m=0.3 | PLDA |  2.77 | 0.172 | 0.271 |
| | | | Cosine | 2.45 | 0.141 | 0.225 |
| config_fbank80_stmn_tseres2net50w26s4_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-Res2Net50 | se-r=256 <br> ArcFace s=30/m=0.3 | PLDA |  2.07 | 0.124 | 0.201 |
| | | | Cosine | 1.95 | 0.113 | 0.181 |
| config_fbank80_stmn_res2net50w13s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=13x8 | ArcFace s=30/m=0.3 | PLDA |  2.34 | 0.136 | 0.230 |
| | | | Cosine | 1.99 | 0.119 | 0.196 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1.sh | Res2Net50 width=26x8 | ArcFace s=30/m=0.3 | PLDA |  2.18 | 0.127 | 0.211 |
| | | | Cosine | 1.89 | 0.112 | 0.184 |
| config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1_swa.sh | Res2Net50 width=26x8 | + SWA | PLDA |  2.14 | 0.125 | 0.209 |
| | | | Cosine | 1.84 | 0.110 | 0.186 |
| config_fbank80_stmn_spinenet49s_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49S | ArcFace s=30/m=0.3 | PLDA | 2.78 | 0.156 | 0.252 |
| | | | Cosine | 2.26 | 0.134 | 0.214 |
| config_fbank80_stmn_spinenet49_arcs30m0.3_adam_lr0.05_amp.v1.sh | SpineNet49 | ArcFace s=30/m=0.3 | Cosine | 2.24 | 0.134 | 0.221 |
| config_fbank80_stmn_spine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 2.20 | 0.132 | 0.219 |
| config_fbank80_stmn_tsespine2net49_arcs30m0.3_adam_lr0.05_amp.v1.sh | Spine2Net49 | ArcFace s=30/m=0.3 | Cosine | 2.02 | 0.123 | 0.203 |

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
   - For better performance use full ResNet (64 base channels) using `config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh` file as
```bash
run_011_train_xvector.sh --config-file config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_030_extract_xvectors.sh --config-file config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh --use-gpu true
run_040_eval_be.sh --config-file config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
```

   - To train with mixed precision training use config file `config_fbank80_mvn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh`

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
| config_fbank80_mvn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.88 | 0.135 | 0.189 |
| | | | Cosine | 1.94 | 0.137 | 0.207 |
| config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA |  1.43 | 0.097 | 0.188 |
| | | | Cosine |  1.35 | 0.088 | 0.146 |
| config_fbank80_mvn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.23 | 0.091 | 0.143 |
| | | | Cosine |  1.17 | 0.081 | 0.110 |



### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | -------------------- | :----: | :------------: | :------------: |
| config_fbank80_mvn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.86 | 0.124 | 0.210 |
| | | | Cosine | 1.91 | 0.120 | 0.208 |
| config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.41 | 0.089 | 0.159 |
| | | | Cosine | 1.19 | 0.078 | 0.140 |
| config_fbank80_mvn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.30 | 0.083 | 0.146 |
| | | | Cosine | 1.09 | 0.071 | 0.124 |


### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | -------------------- | :----: | :------------: | :------------: |
| config_fbank80_mvn_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | LResNet34 | ArcFace s=30/m=0.3 | PLDA | 3.24 | 0.197 | 0.320 |
| | | | Cosine | 3.21 | 0.190 | 0.302 |
| config_fbank80_mvn_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.65 | 0.155 | 0.254 |
| | | | Cosine | 2.27 | 0.137 | 0.219 |
| config_fbank80_mvn_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.46 | 0.142 | 0.237 |
| | | | Cosine |  2.14 | 0.126 | 0.203 |


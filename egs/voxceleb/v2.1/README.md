# VoxCeleb V2.1

Recipe for the VoxCeleb Speaker Verification Task using Wav2Vec2, WavLM or Hubert models from HuggingFace as feature extractors

## Differences w.r.t VoxCeleb V2 recipe

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
   - By default it will use config global_conf/config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh
   - To use other configs: 
```bash
run_005_train_xvector.sh --config-file global_conf/other_config.sh
run_006_extract_xvectors.sh --config-file global_conf/other_config.sh --use-gpu true
run_007_eval_be.sh --config-file global_conf/other_config.sh
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

   - `run_004_prepare_xvec_train_data.sh`
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_005_train_xvector.sh`
      - Trains the x-vector model on frozen wav2vec features
      - Finetunes wav2vec+x-vector model
      - Large margin finetuning of wav2vec+x-vector model

   - `run_006_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Exctracts x-vectors for VoxCeleb1 test sets

   - `run_007_eval_be.sh`
      - Trains PLDA and evals PLDA and cosine scoring back-ends


## Results





### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.84 | 0.060 | 0.116 |
| | | | Cosine + AS-Norm | 0.81 | 0.058 | 0.108 |
| | | | Cosine + QMF | 0.75 | 0.054 | 0.086 |
| config_wavlmbaseplus9l_ecapatdnn512x3_v2.0.sh | WavLM(layer=2-9)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.89 | 0.069 | 0.108 |
| | | | Cosine + AS-Norm | 0.86 | 0.067 | 0.108 |
| | | | Cosine + QMF | 0.77 | 0.066 | 0.105 |
| config_wavlmlarge_ecapatdnn512x3_v2.0.sh | WavLM-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.74 | 0.057 | 0.085 |
| | | | Cosine + AS-Norm | 0.73 | 0.055 | 0.093 |
| | | | Cosine + QMF | 0.66 | 0.051 | 0.094 |
| config_wavlmlarge12l_ecapatdnn512x3_v2.0.sh | WavLM-Large(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.74 | 0.053 | 0.080 |
| | | | Cosine + AS-Norm | 0.71 | 0.050 | 0.087 |
| | | | Cosine + QMF | 0.64 | 0.045 | 0.087 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.84 | 0.063 | 0.111 |
| | | | Cosine + AS-Norm | 0.68 | 0.053 | 0.090 |
| | | | Cosine + QMF | 0.63 | 0.048 | 0.071 |
| config_wav2vec2xlsr300m12l_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.14 | 0.074 | 0.107 |
| | | | Cosine + AS-Norm | 0.94 | 0.060 | 0.089 |
| | | | Cosine + QMF | 0.89 | 0.054 | 0.076 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.1.sh | Wav2Vec2-XLSR300M(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.69 | 0.048 | 0.094 |
| | | | Cosine + AS-Norm | 0.63 | 0.046 | 0.082 |
| | | | Cosine + QMF | 0.57 | 0.041 | 0.076 |

### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.81 | 0.051 | 0.087 |
| | | | Cosine + AS-Norm | 0.78 | 0.047 | 0.083 |
| | | | Cosine + QMF | 0.75 | 0.046 | 0.076 |
| config_wavlmbaseplus9l_ecapatdnn512x3_v2.0.sh | WavLM(layer=2-9)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.89 | 0.056 | 0.099 |
| | | | Cosine + AS-Norm | 0.86 | 0.053 | 0.090 |
| | | | Cosine + QMF | 0.82 | 0.050 | 0.085 |
| config_wavlmlarge_ecapatdnn512x3_v2.0.sh | WavLM-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.80 | 0.049 | 0.088 |
| | | | Cosine + AS-Norm | 0.76 | 0.045 | 0.080 |
| | | | Cosine + QMF | 0.73 | 0.043 | 0.078 |
| config_wavlmlarge12l_ecapatdnn512x3_v2.0.sh | WavLM-Large(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.91 | 0.056 | 0.094 |
| | | | Cosine + AS-Norm | 0.87 | 0.053 | 0.090 |
| | | | Cosine + QMF | 0.83 | 0.050 | 0.086 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.80 | 0.050 | 0.086 |
| | | | Cosine + AS-Norm | 0.73 | 0.045 | 0.074 |
| | | | Cosine + QMF | 0.69 | 0.042 | 0.069 |
| config_wav2vec2xlsr300m12l_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M(layer=2-12)-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.99 | 0.058 | 0.103 |
| | | | Cosine + AS-Norm | 0.87 | 0.052 | 0.090 |
| | | | Cosine + QMF | 0.83 | 0.050 | 0.085 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.1.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 0.72 | 0.044 | 0.079 |
| | | | Cosine + AS-Norm | 0.68 | 0.040 | 0.068 |
| | | | Cosine + QMF | 0.64 | 0.037 | 0.065 |

### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.73 | 0.113 | 0.182 |
| | | | Cosine + AS-Norm | 1.63 | 0.100 | 0.160 |
| | | | Cosine + QMF | 1.56 | 0.096 | 0.155 |
| config_wavlmbaseplus9l_ecapatdnn512x3_v2.0.sh | WavLM(layer=2-9)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.88 | 0.122 | 0.200 |
| | | | Cosine + AS-Norm | 1.77 | 0.110 | 0.175 |
| | | | Cosine + QMF | 1.66 | 0.104 | 0.168 |
| config_wavlmlarge_ecapatdnn512x3_v2.0.sh | WavLM-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.67 | 0.103 | 0.165 |
| | | | Cosine + AS-Norm | 1.54 | 0.093 | 0.152 |
| | | | Cosine + QMF | 1.45 | 0.089 | 0.145 |
| config_wavlmlarge12l_ecapatdnn512x3_v2.0.sh | WavLM-Large(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.78 | 0.106 | 0.174 |
| | | | Cosine + AS-Norm | 1.70 | 0.099 | 0.162 |
| | | | Cosine + QMF | 1.61 | 0.094 | 0.153 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.49 | 0.087 | 0.137 |
| | | | Cosine + AS-Norm | 1.29 | 0.074 | 0.117 |
| | | | Cosine + QMF | 1.22 | 0.069 | 0.111 |
| config_wav2vec2xlsr300m12l_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M(layer=2-12)-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.84 | 0.107 | 0.172 |
| | | | Cosine + AS-Norm | 1.47 | 0.083 | 0.128 |
| | | | Cosine + QMF | 1.39 | 0.079 | 0.123 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 1.24 | 0.076 | 0.121 |
| | | | Cosine + AS-Norm | 1.15 | 0.068 | 0.109 |
| | | | Cosine + QMF | 1.09 | 0.065 | 0.107 |

### VoxSRC2022 dev

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh | WavLM+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.60 | 0.163 | 0.257 |
| | | | Cosine + AS-Norm | 2.43 | 0.150 | 0.244 |
| | | | Cosine + QMF | 2.31 | 0.143 | 0.232 |
| config_wavlmbaseplus9l_ecapatdnn512x3_v2.0.sh | WavLM(layer=2-9)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.82 | 0.183 | 0.286 |
| | | | Cosine + AS-Norm | 2.69 | 0.168 | 0.265 |
| | | | Cosine + QMF | 2.52 | 0.158 | 0.252 |
| config_wavlmlarge_ecapatdnn512x3_v2.0.sh | WavLM-Large+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.65 | 0.176 | 0.289 |
| | | | Cosine + AS-Norm | 2.55 | 0.171 | 0.292 |
| | | | Cosine + QMF | 2.38 | 0.159 | 0.266 |
| config_wavlmlarge12l_ecapatdnn512x3_v2.0.sh | WavLM-Large(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.62 | 0.153 | 0.251 |
| | | | Cosine + AS-Norm | 2.53 | 0.149 | 0.247 |
| | | | Cosine + QMF | 2.42 | 0.144 | 0.231 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.25 | 0.136 | 0.225 |
| | | | Cosine + AS-Norm | 2.01 | 0.125 | 0.209 |
| | | | Cosine + QMF | 1.92 | 0.117 | 0.200 |
| config_wav2vec2xlsr300m12l_ecapatdnn512x3_v2.0.sh | Wav2Vec2-XLSR300M(layer=2-12)+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.83 | 0.175 | 0.276 |
| | | | Cosine + AS-Norm | 2.31 | 0.149 | 0.244 |
| | | | Cosine + QMF | 2.22 | 0.137 | 0.229 |
| config_wav2vec2xlsr300m_ecapatdnn512x3_v2.1.sh | Wav2Vec2-XLSR300M+ECAPA-TDNN 512x3 | Stage3: ArcFace m=0.4/intertop_m=0.1 | Cosine | 2.06 | 0.124 | 0.206 |
| | | | Cosine + AS-Norm | 1.97 | 0.125 | 0.212 |
| | | | Cosine + QMF | 1.87 | 0.120 | 0.204 |


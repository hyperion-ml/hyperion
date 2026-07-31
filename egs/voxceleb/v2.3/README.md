# VoxCeleb V2.3

Recipe for the VoxCeleb Speaker Verification Task using Wav2Vec2, WavLM, or
HuBERT models from Hugging Face as front-ends for x-vector networks.

## What this recipe does

This version uses Hyperion table formats and command-line entry points. The
front-end is loaded from Hugging Face and is combined with an ECAPA-TDNN
x-vector network. Training includes frozen-front-end training followed by
fine-tuning and optional large-margin objectives, depending on the selected
configuration.

The recipe no longer uses Kaldi queue wrappers. Jobs are launched through
``hyperion-submit`` and the backend is selected in ``cmd.sh``.

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

Run the stages from the recipe directory in order. The default model
configuration is
``global_conf/config_wavlmbaseplus_ecapatdnn512x3_v2.0.sh``.

To select another front-end or network configuration:

```bash
./run_005_train_xvector.sh --config-file global_conf/other_config.sh
./run_006_extract_xvectors.sh --config-file global_conf/other_config.sh --use-gpu true
./run_007_eval_be.sh --config-file global_conf/other_config.sh
```

Each stage supports its usual ``stage`` and configuration options, so a
completed stage can be skipped when resuming an experiment.

## Job submission

``cmd.sh`` selects the backend for all stages. On the configured Slurm site it
uses ``conf/submit_coe_v100.yaml``; on other hosts it uses the synchronous
local backend. The YAML file contains site-specific ``sbatch`` policy, while
the command variables in ``cmd.sh`` request portable resources:

* ``train_cmd``: CPU jobs with 4 GB of memory.
* ``cuda_cmd``: GPU training jobs with 30 GB of memory.
* ``cuda_eval_cmd``: GPU evaluation jobs with 4 GB of memory.

The local backend uses the current Python environment and runs the command
synchronously:

```bash
hyperion-submit local --output-file exp/example/log/job.log -- \
  hyperion-eval-verification-metrics --help
```

To submit directly to Slurm, select the recipe's YAML policy and request the
resources needed by the command:

```bash
hyperion-submit slurm --cfg conf/submit_coe_v100.yaml \
  --num-gpus 4 --num-threads 9 --mem 30G \
  --output-file exp/example/log/train.log -- \
  hyperion-train-wav2vec2xvector ...
```

When more than one GPU is requested, ``hyperion-submit`` prepends
``torchrun`` and preserves Slurm's ``CUDA_VISIBLE_DEVICES`` assignment. It
does not search for or select GPUs itself.

Large extraction and preprocessing stages use arrays. ``JOB`` is substituted
in command arguments and output paths, and ``--max-jobs-run`` can limit the
number of simultaneously running tasks:

```bash
hyperion-submit slurm --cfg conf/submit_coe_v100.yaml \
  --array JOB=1:100 --max-jobs-run 20 --num-gpus 1 \
  --output-file exp/example/log/extract.JOB.log -- \
  hyperion-extract-wav2vec2xvectors --part-idx JOB --num-parts 100
```

Submission is synchronous: the stage waits for every local or Slurm task and
returns a failure if any task fails. Slurm fallback diagnostics are kept in
the ``q/`` directory beside the requested log.


## Recipe Steps:

   - `run_001_prepare_data.sh`
      - Generates Hyperion tables for VoxCeleb2 train/test and VoxCeleb1 O/E/H evaluation sets.

   - `run_002_compute_evad.sh`
      - Computes energy-based VAD for all datasets, using a synchronous array of jobs.

   - `run_003_prepare_noises_rirs.sh`
      - Prepares MUSAN noise and music for SpeechAugment.
      - Creates babble noise from MUSAN speech.
      - Packs RIRS_NOISES into HDF5-backed Hyperion datasets.

   - `run_004_prepare_xvec_train_data.sh`
      - Converts training audio to a common format and applies optional VAD trimming.
      - Removes short utterances and speakers with too few samples.
      - Creates training and validation tables for x-vector training.

   - `run_005_train_xvector.sh`
      - Trains the x-vector model with the selected frozen front-end.
      - Fine-tunes the front-end and x-vector network.
      - Optionally applies a large-margin objective.

   - `run_006_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Extracts x-vectors for VoxCeleb1 evaluation sets.

   - `run_007_eval_be.sh`
      - Trains and evaluates cosine, AS-Norm, QMF, and PLDA back ends.


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

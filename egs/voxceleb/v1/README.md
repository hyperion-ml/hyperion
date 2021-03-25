# VoxCeleb Version 1

Last update 2020/08/05

Recipe for the VoxCeleb Speaker Verification Task using several flavors of x-Vectors

This recipe computes acoustic features and speech augmentations off-line.
Look version V1.1, for a newer recipe which computes features 
and augmentations on the fly.

## Training Data

   - x-Vector network is trained on Voxceleb2 dev + test with augmentations
     - MUSAN noise
     - RIR reverberation

## Test Data

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
   - You can skip the x-vector finetuning scripts since they don't improve the results (steps 12, 31 and 41)
   - By default it will use Light ResNet (16 channels)
   - For better performance use full ResNet (64 channels) using `global_conf/config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh` file as
```bash
run_011_train_xvector.sh --config-file global_conf/config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_030_extract_xvectors.sh --config-file global_conf/config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
run_040_eval_be.sh --config-file global_conf/config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh
```

   - The `amp` suffix in the config files means that we train with mixed precission to reduce GPU memory requirements.

## Recipe Steps:

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
          - VoxCeleb2 train+test
          - VoxCeleb1 O/E/H eval sets

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_003_compute_fbank.sh`
      - Computes log-filter-banks acoustic features for all datasets

   - `run_004_prepare_augment.sh`
      - Prepares Kaldi style data directories for augmented training data with MUSAN noise and RIR reverberation.

   - `run_005_compute_fbank_augment.sh
      - Computes log-filter-banks for augmented datasets

   - `run_010_prepare_xvec_train_data.sh`
      - Prepares features for x-vector training
      - Applies sort-time mean normalization and remove silence frames
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_xvector.sh`
      - Trains the x-vector network

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for VoxCeleb2 or VoxCeleb2+augmentation for PLDA training
      - Exctracts x-vectors for VoxCeleb1 test sets

   - `run_040_eval_be.sh`
      - Trains PLDA and evals PLDA and cosine scoring back-ends
      - Results are left in `exp/scores/..../voxceleb1_{o,e,h}_clean_results` files


## Results

### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_resetdnn_narrow_arcs30s0.3_adam_lr0.05_amp.v1.sh | ResETDNN | num-blocks=5 / hid-channels=512 <br> ArcFace s=30/m=0.3 | PLDA | 2.80 | 0.178 | 0.283 |
| | | | Cosine | 3.03 | 0.201 | 0.301 |
| config_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Light ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.01 | 0.149 | 0.213 |
| | | | Cosine | 2.23 | 0.153 | 0.231 |
| config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.42 | 0.102 | 0.174 |
| | | | Cosine | 1.31 | 0.089 | 0.135 |
| config_resnet34_arcs30m0.3_adam_lr0.05_wo_aug_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 <br> without augmentation | PLDA | 1.48 | 0.103 | 0.180 |
| | | | Cosine | 1.54 | 0.097 | 0.153 |
| config_seresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.39 | 0.094 | 0.152 |
| | | | Cosine | 1.27 | 0.078 | 0.120 |
| config_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.27 | 0.085 | 0.127 |
| | | | Cosine | 1.19 | 0.076 | 0.109 |
| config_effnetb0_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA | 2.30 | 0.169 | 0.260 |
| | | | Cosine | 2.05 | 0.142 | 0.186 |
| config_effnetb0_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA | 1.61 | 0.131 | 0.228 |
| | | | Cosine | 1.54 | 0.095 | 0.165 |
| config_effnetb4_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA | 2.01 | 0.166 | 0.276 |
| | | | Cosine | 1.71 | 0.124 | 0.207 |
| config_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA | 1.49 | 0.113 | 0.205 |
| | | | Cosine | 1.29 | 0.084 | 0.149 |
| config_transformer_lac6b6d512h8ff2048_arcs30m0.3_adam_lr0.005_amp.v1.sh | Transformer | Att-context=6 / blocks=6 <br> d_model=512/ heads=8 / d_ff=2048 <br> ArcFace s=30/m=0.3 | PLDA | 1.87 | 0.136 | 0.212 |
| | | | Cosine | 2.18 | 0.138 | 0.224 |


### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | --------------| --------- | :----: | :------------: | :------------: |
| config_resetdnn_narrow_arcs30s0.3_adam_lr0.05_amp.v1.sh | ResETDNN | num-blocks=5 / hid-channels=512 <br> ArcFace s=30/m=0.3 | PLDA |2.77 | 0.183 | 0.298 |
| | | | Cosine | 3.23 | 0.206 | 0.336 |
| config_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Light ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.99 | 0.132 | 0.226 |
| | | | Cosine | 2.10 | 0.134 | 0.220 |
| config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.46 | 0.097 | 0.168 |
| | | | Cosine | 1.33 | 0.087 | 0.152 |
| config_resnet34_arcs30m0.3_adam_lr0.05_wo_aug_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 <br> without augmentation | PLDA | 1.57 | 0.102 | 0.183 |
| | | | Cosine | 1.47 | 0.096 | 0.166 |
| config_seresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA |1.55 | 0.099 | 0.168 |
| | | | Cosine | 1.33 | 0.083 | 0.140 |
| config_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 1.36 | 0.089 | 0.155 |
| | | | Cosine | 1.20 | 0.078 | 0.136 |
| config_effnetb0_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA | 2.36 | 0.163 | 0.278 |
| | | | Cosine | 2.07 | 0.135 | 0.227 |
| config_effnetb0_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA |1.72 | 0.117 | 0.206 |
| | | | Cosine | 1.46 | 0.095 | 0.167 |
| config_effnetb4_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA | 2.25 | 0.150 | 0.256 |
| | | | Cosine |1.77 | 0.117 | 0.206 |
| config_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA | 1.67 | 0.109 | 0.197 |
| | | | Cosine | 1.27 | 0.083 | 0.148 |
| config_transformer_lac6b6d512h8ff2048_arcs30m0.3_adam_lr0.005_amp.v1.sh | Transformer | Att-context=6 / blocks=6 <br> d_model=512/ heads=8 / d_ff=2048 <br> ArcFace s=30/m=0.3 | PLDA |1.95 | 0.129 | 0.220 |
| | | | Cosine |2.08 | 0.131 | 0.214 |


### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | Model Details | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | -------- | :----: | :------------: | :------------: |
| config_resetdnn_narrow_arcs30s0.3_adam_lr0.05_amp.v1.sh | ResETDNN | num-blocks=5 / hid-channels=512 <br> ArcFace s=30/m=0.3 | PLDA | 4.86 | 0.290 | 0.441 |
| | | | Cosine |5.47 | 0.310 | 0.458 |
| config_lresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Light ResNet34 | ArcFace s=30/m=0.3 | PLDA |3.51 | 0.214 | 0.335 |
| | | | Cosine | 3.52 | 0.203 | 0.321 |
| config_resnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.77 | 0.165 | 0.268 |
| | | | Cosine |2.46 | 0.147 | 0.242 |
| config_resnet34_arcs30m0.3_adam_lr0.05_wo_aug_amp.v1.sh | ResNet34 | ArcFace s=30/m=0.3 <br> without augmentation | PLDA |2.87 | 0.173 | 0.282 |
| | | | Cosine |2.65 | 0.156 | 0.252 |
| config_seresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.85 | 0.168 | 0.271 |
| | | | Cosine |2.48 | 0.148 | 0.237 |
| config_tseresnet34_arcs30m0.3_adam_lr0.05_amp.v1.sh | Time-SE-ResNet34 | ArcFace s=30/m=0.3 | PLDA | 2.58 | 0.153 | 0.243 |
| | | | Cosine | 2.29 | 0.135 | 0.223 |
| config_effnetb0_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA |4.25 | 0.253 | 0.394 |
| | | | Cosine | 3.56 | 0.207 | 0.327 |
| config_effnetb0_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b0 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA | 3.05 | 0.185 | 0.300 |
| | | | Cosine | 2.66 | 0.154 | 0.249 |
| config_effnetb4_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=16 <br> ArcFace s=30/m=0.3 | PLDA |4.06 | 0.244 | 0.377 |
| | | | Cosine |3.21 | 0.192 | 0.301 |
| config_effnetb4_v2_arcs30m0.3_adam_lr0.01_amp.v1.sh | EfficientNet-b4 | Enc Downsampling=8 <br> ArcFace s=30/m=0.3 | PLDA | 2.94 | 0.176 | 0.281 |
| | | | Cosine | 2.28 | 0.136 | 0.224 |
| config_transformer_lac6b6d512h8ff2048_arcs30m0.3_adam_lr0.005_amp.v1.sh | Transformer | Att-context=6 / blocks=6 <br> d_model=512/ heads=8 / d_ff=2048 <br> ArcFace s=30/m=0.3 | PLDA | 3.42 | 0.209 | 0.333 |
| | | | Cosine |3.51 | 0.206 | 0.323 |

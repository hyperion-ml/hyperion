# VoxCeleb SSL V1

Recipe for the Unsupervised VoxCeleb Speaker Verification Task:
  - Trains embedding extractor using DINO
  - Cluster embeddings of VoxCeleb2 to get pseuso-speaker labels
  - Embedding Model is fine-tuned with Large Margin Softmax loss on the pseudo-speaker labels
  - Repeakt embedding clustering to get new pseuso-speaker labels
  - Embedding Model is fine-tuned with Large Margin Softmax loss on the new pseudo-speaker labels
 
## Citing

If you use our DINO implementation, please, cite these works:

```
@ARTICLE{9852303,
  author={Cho, Jaejin and Villalba, Jesús and Moro-Velazquez, Laureano and Dehak, Najim},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Non-Contrastive Self-Supervised Learning for Utterance-Level Information Extraction From Speech}, 
  year={2022},
  volume={16},
  number={6},
  pages={1284-1295},
  keywords={Alzheimer's disease;Transfer learning;Speech processing;Feature extraction;Self-supervised learning;Training;Emotion recognition;Self-supervised learning;transfer learning;speaker verification;emotion recognition;Alzheimer's disease;distillation;non-contrastive},
  doi={10.1109/JSTSP.2022.3197315}}

@inproceedings{cho22c_interspeech,
  author={Jaejin Cho and Raghavendra Pappagari and Piotr Żelasko and Laureano Moro Velazquez and Jesus Villalba and Najim Dehak},
  title={{Non-contrastive self-supervised learning of utterance-level speech representations}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4028--4032},
  doi={10.21437/Interspeech.2022-11141}
}
```

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
   - By default it will use config global_conf/config_fbank80_stmn_fwseresnet34.v1.2.1.sh
   - To use other configs: 
```bash
run_xxx_xxxx.sh --config-file global_conf/other_config.sh
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

    - `run_005_train_dino.sh`
      - Trains DINO embeddings

    - `run_006_extract_dino_embeds_cluster_eval.sh`
      - Extracts DINO embeddings for Vox2 and Vox1
      - Evaluates SV metrics in Vox1-O/E/H using Cosine Scoring
      - Clusters Vox2 Embeddings into pseudo-speakers
      - Trains PLDA on Vox2 pseudo-speakers
      - Evaluates SV metrics in Vox1-O/E/H using PLDA

   - `run_007_train_xvector.sh`
      - Fine-tunes DINO model in x-vector style using pseudo-labels from previous step
      - First, it finetunes x-vector projection and output layer with the rest of network frozen
      - Second, it finetunes full network

    - `run_008_extract_ft1_xvec_embeds_cluster_eval.sh`
      - Extracts X-Vector embeddings for Vox2 and Vox1
      - Evaluates SV metrics in Vox1-O/E/H using Cosine Scoring
      - Clusters Vox2 Embeddings into pseudo-speakers
      - Trains PLDA on Vox2 pseudo-speakers
      - Evaluates SV metrics in Vox1-O/E/H using PLDA
      
    - `run_009_finetune_xvector_s2.sh`
      - Fine-tunes X-Vector model in x-vector style using pseudo-labels from previous step
      - First, it finetunes x-vector projection and output layer with the rest of network frozen
      - Second, it finetunes full network
    
    - `run_010_extract_ft2_xvec_embeds_cluster_eval.sh`
      - Extracts X-Vector embeddings for Vox2 and Vox1
      - Evaluates SV metrics in Vox1-O/E/H using Cosine Scoring
      - Clusters Vox2 Embeddings into pseudo-speakers
      - Trains PLDA on Vox2 pseudo-speakers
      - Evaluates SV metrics in Vox1-O/E/H using PLDA


## Results

### VoxCeleb 1 Original-Clean trial list

| Config | Model Type | DINO Clustering | X-Vector Clustering | Stage | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | --------------- | ------------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34.v1.2.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 3.96 | 0.276 | 0.423 |
| | | | | | PLDA | 3.18 | 0.182 | 0.273 |
| | | | | FT-1 | Cosine | 1.97 | 0.139 | 0.214 |
| | | | | FT-2 | Cosine | 1.80 | 0.133 | 0.200 |
| config_fbank80_stmn_lresnet34.v1.2.1.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 1.75 | 0.124 | 0.197 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.sh | ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 4.14 | 0.274 | 0.405 |
| | | | | | PLDA | 4.16 | 0.225 | 0.361 |
| | | | | FT-1 | Cosine | 2.68 | 0.173 | 0.258 |
| | | | | FT-2 | Cosine | 2.57 | 0.151 | 0.244 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.1.sh| ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 2.71 | 0.169 | 0.243 |
| config_fbank80_stmn_fwseresnet34.v1.2.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 4.57 | 0.344 | 0.553 |
| | | | | | PLDA | 2.92 | 0.232 | 0.410 |
| | | | | FT-1 | Cosine | 2.11 | 0.135 | 0.223 |
| | | | | FT-1 | PLDA | 1.75 | 0.137 | 0.236 |
| | | | | FT-2 | Cosine | 1.65 | 0.116 | 0.168 |
| | | | | FT-2 | PLDA | 1.67 | 0.137 | 0.193 |
| config_fbank80_stmn_fwseresnet34.v1.2.1.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 1.49 | 0.101 | 0.161 |
| | | | | FT-2 | PLDA | 1.53 | 0.109 | 0.168|
| config_fbank80_stmn_fwseresnet34.v1.2.2.sh  | FW-SE ResNet34 / 0.1 x Cos Reg. | Cos+AHC+PLDA+AHC | Cos+AHC | DINO | Cosine | 3.96 | 0.232 | 0.358 |
| | | | | | PLDA | 4.04 | 0.185 | 0.291 |
| | | | | FT-1 | Cosine | 2.03 | 0.125 | 0.203 |
| | | | | FT-1 | PLDA | 2.44 | 0.149 | 0.231 |
| | | | | FT-2 | Cosine | 
| | | | | FT-2 | PLDA | 


### VoxCeleb 1 Entire-Clean trial list

| Config | Model Type | DINO Clustering | X-Vector Clustering | Stage | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | --------------- | ------------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34.v1.2.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 4.94 | 0.304 | 0.483 |
| | | | | | PLDA | 3.72 | 0.184 | 0.300 |
| | | | | FT-1 | Cosine | 2.35 | 0.136 | 0.217 |
| | | | | FT-2 | Cosine | 2.02 | 0.118 | 0.195 |
| config_fbank80_stmn_lresnet34.v1.2.1.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 1.98 | 0.116 | 0.185 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.sh | ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 4.61 | 0.293 | 0.455|
| | | | | | PLDA | 3.91 | 0.223 | 0.356 |
| | | | | FT-1 | Cosine | 3.04 | 0.168 | 0.263 |
| | | | | FT-2 | Cosine | 2.83 | 0.155 | 0.248 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.1.sh| ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 3.06 | 0.164 | 0.256 |
| config_fbank80_stmn_fwseresnet34.v1.2.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 5.50 | 0.426 | 0.664 |
| | | | | | PLDA | 3.33 | 0.245 | 0.425 |
| | | | | FT-1 | Cosine | 2.42 | 0.147 | 0.243 |
| | | | | FT-1 | PLDA | 2.03 | 0.144 | 0.255 |
| | | | | FT-2 | Cosine | 1.86 | 0.112 | 0.186 |
| | | | | FT-2 | PLDA | 1.77 | 0.121 | 0.208 |
| config_fbank80_stmn_fwseresnet34.v1.2.1.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 1.83 | 0.106 | 0.170 |
| | | | | FT-2 | PLDA | 1.68 | 0.109 | 0.188 |
| config_fbank80_stmn_fwseresnet34.v1.2.2.sh  | FW-SE ResNet34 / 0.1 x Cos Reg. | Cos+AHC+PLDA+AHC | Cos+AHC | DINO | Cosine | 4.31 | 0.250 | 0.387 |
| | | | | | PLDA | 4.32 | 0.166 | 0.263 |
| | | | | FT-1 | Cosine | 2.61 | 0.138 | 0.210 | 
| | | | | FT-1 | PLDA | 2.72 | 0.1366 | 0.216 |
| | | | | FT-2 | Cosine | 
| | | | | FT-2 | PLDA | 


### VoxCeleb 1 Hard-Clean trial list

| Config | Model Type | DINO Clustering | X-Vector Clustering | Stage | Back-end | EER(%) | MinDCF(p=0.05) | MinDCF(p=0.01) |
| ------ | ---------- | ------------- | --------------- | ------------------- | -------- | :----: | :------------: | :------------: |
| config_fbank80_stmn_lresnet34.v1.2.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 8.33 | 0.462 | 0.664 |
| | | | | | PLDA | 5.91 | 0.304 | 0.481 |
| | | | | FT-1 | Cosine | 3.89 | 0.215 | 0.340 |
| | | | | FT-2 | Cosine | 3.44 | 0.192 | 0.303 |
| config_fbank80_stmn_lresnet34.v1.2.1.sh | LResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 3.33 | 0.185 | 0.290 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.sh | ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 8.38 | 0.458 | 0.635 |
| | | | | | PLDA | 6.48 | 0.360 | 0.532 |
| | | | | FT-1 | Cosine | 4.93 | 0.259 | 0.383 |
| | | | | FT-2 | Cosine | 4.73 | 0.251 | 0.375 |
| config_fbank80_stmn_ecapatdnn512x3.v1.2.1.sh| ECAPA-TDNN 512x3 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 4.90 | 0.251 | 0.378 |
| config_fbank80_stmn_fwseresnet34.v1.2.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC+PLDA+AHC | DINO | Cosine | 10.9 | 0.644 | 0.822 |
| | | | | | PLDA | 6.86 | 0.481 | 0.745 |
| | | | | FT-1 | Cosine | 4.35 | 0.25 | 0.393 |
| | | | | FT-1 | PLDA | 4.21 | 0.281 | 0.452
| | | | | FT-2 | Cosine | 3.37 | 0.194 | 0.309 |
| | | | | FT-2 | PLDA | 3.51 | 0.219 | 0.351 |
| config_fbank80_stmn_fwseresnet34.v1.2.1.sh  | FW-SE ResNet34 | Cos+AHC+PLDA+AHC | Cos+AHC | FT-2 | Cosine | 3.11 | 0.172 | 0.270 |
| | | | | FT-2 | PLDA | 3.15 | 0.186 | 0.294 |
| config_fbank80_stmn_fwseresnet34.v1.2.2.sh  | FW-SE ResNet34 / 0.1 x Cos Reg. | Cos+AHC+PLDA+AHC | Cos+AHC | DINO | Cosine | 7.41 | 0.377 | 0.526 |
| | | | | | PLDA | 5.95 | 0.269 | 0.438 |
| | | | | FT-1 | Cosine | 4.38 | 0.222 | 0.337 |
| | | | | FT-1 | PLDA | 4.68 | 0.237 | 0.375 |
| | | | | FT-2 | Cosine | 
| | | | | FT-2 | PLDA | 


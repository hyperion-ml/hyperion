# SRE24 Audio V1 Open

x-Vector recipe for SRE24 audio open condition
The systems runs at 16 kHz, telephone data is upsampled to 16k

## Citing

   This recipe is based on these works
```
@inproceedings{Villalba2022,
author = {Jes\'us Villalba and Bengt J Borgstrom and Saurabh Kataria and Magdalena Rybicka and Carlos D Castillo and Jaejin Cho and L. Paola García-Perera and Pedro A. Torres-Carrasquillo and Najim Dehak},
city = {ISCA},
doi = {10.21437/Odyssey.2022-30},
issue = {July},
journal = {The Speaker and Language Recognition Workshop (Odyssey 2022)},
month = {6},
pages = {213-220},
publisher = {ISCA},
title = {Advances in Cross-Lingual and Cross-Source Audio-Visual Speaker Recognition: The JHU-MIT System for NIST SRE21},
url = {https://www.isca-speech.org/archive/odyssey_2022/villalba22b_odyssey.html},
year = {2022},
}
				 
@inproceedings{Villalba2020,
address = {Tokyo, Japan},
author = {Villalba, Jes{\'{u}}s and Garcia-Romero, Daniel and Chen, Nanxin and Sell, Gregory and Borgstrom, Jonas and McCree, Alan and {Garcia Perera}, Leibny Paola and Kataria, Saurabh and Nidadavolu, Phani Sankar and Torres-Carrasquiilo, Pedro and Dehak, Najim},
booktitle = {Odyssey 2020 The Speaker and Language Recognition Workshop},
doi = {10.21437/Odyssey.2020-39},
month = {nov},
pages = {273--280},
title = {{Advances in Speaker Recognition for Telephone and Audio-Visual Data: the JHU-MIT Submission for NIST SRE19}},
url = {http://www.isca-speech.org/archive/Odyssey{\_}2020/abstracts/88.html},
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
volume = {60},
year = {2020}
}
```

## Training Data
   - x-Vector networks and PLDA are trained on
     - SRE CTS Superset
     - SRE16 Eval
     - SRE21 Eval
     - SRE18 CMN2
     - SRE19 CTS
     - Voxceleb 1+2
     with augmentations:
     - MUSAN noise
     - RIR reverberation
     - Telephone Codecs

## Adaptation Data
   - PLDA is adapted on SRE24 Dev
     - Using 2 fold cross-validation to Evaluate on SRE24 Dev
     - Full SRE24 Dev to Evaluate on SRE24 Eval

## Test data

   We evaluate:
     - SRE24 Dev 2 Fold cross-val
     - SRE24 Dev Full adapted on itself (cheating)
     - SRE24 Eval

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it uses ResNet34 x-vector
   - To choose other network use config files as
```bash
run_0xx_....sh --config-file global_conf/config_fbank80_stmn_res2net50w26s8_arcs30m0.3_adam_lr0.05_amp.v1.sh
```

## Recipe Steps

   - `run_001_prepare_data.sh`
     - Data preparation script to generate Hyperion style data directories for 
       - SRE Superset
       - SRE16 Eval
       - SRE21 Dev/Eval
       - SRE24 Dev/Eval

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
      - Trains the x-vector network on 4sec chunks
      - Fine-tune x-vector network on 10-15 secs utts

   - `run_006_eval_lid.sh`
      - Evaluates LID using ResNet100 model from LRE22 recipe on:
        - SRE24 Eval
      
   - `run_007_extract_xvectors.sh`
      - Computes x-vectors for all datasets

   - `run_008_eval_be_v1.sh`
      - Train/Evals back-end: Centering + PCA + LNorm + PLDA, Centering adapted to source and language, global PLDA adapted to SRE-Vox-CHN

   - `run_009_extract_xvectors_diarization.sh`
      - Compute x-vectors for SRE24 Dev/Eval test with diarization

   - `run_010_eval_be_v1_diarization.sh`
      - Evals back-end on SRE24 Dev/Eval using the diarized x-vectors

   - `run_011_fusion_example_v1.sh`
      - Fusion example
      - You need the scores of all systems for this to work

## Results

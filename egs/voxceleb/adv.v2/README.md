# VoxCeleb Adversarial Attacks Version 2

Last update 2021/05/17

Recipe to evaluate Adversarial attacks classification, verification and novel attack detection on
attacks against speaker identification and speaker verification.

## Citing

  If you use this recipe, please cite:
```
@inproceedings{Villalba2021,
address = {Brno, Czech Republic},
author = {Villalba, Jes{\'{u}}s and Joshi, Sonal and Zelasko, Pietr and Dehak, Najim},
booktitle = {Interspeech 2021},
month = {sep},
title = {{Representation Learning to Classify and Detect Adversarial Attacks against Speaker and Speech Recognition Systems}},
year = {2021}
}
```


## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it will use Light ResNet34 as speaker id victim model and signature extraction model and will 
     train and test on successful attacks with all SNRs
   - You can change that modifying the configuration script.
   - For example, to use Res2Net as attack signature model use `config_spknet_fbank80_stmn_lresnet34_attacknet_res2net.v1.sh` 
     when calling each of the steps as
```bash
run_0*.sh --config-file global_conf/config_spknet_fbank80_stmn_lresnet34_attacknet_res2net.v1.sh
```
   - To use just attacks with SNR>20 dB use `global_conf/config_spknet_fbank80_stmn_lresnet34_attacknet_same_snrge20dB.v1.sh`

## Recipe Steps:
   First steps are to train victim x-vector network and are similar to other recipes

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
          - VoxCeleb2 train+test
          - VoxCeleb1 Original eval sets

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_002b_compute_fbank.sh`
      - Computes log-filter-banks acoustic features for all datasets

   - `run_003_prepare_noises_rirs.sh`
      - Prepares MUSAN noises, music to be used by SpeechAugment class.
      - Creates Babble noise from MUSAN speech to be used by SpeechAugment class.
      - Prepares RIRs by compacting then into HDF5 files, to be used by SpeechAugment class.

   - `run_010_prepare_xvec_train_data.sh`
      - Prepares audios train the victim x-vector model
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_victim_xvector.sh`
      - Trains the victim x-vector network

   - `run_020_generate_identif_attacks.sh`
      - Generate attacks against speaker identification task using hyperion attack classes

   - `run_021_split_train_test.sh`
      - Splits generated attacks into training/validation/test
      - Creates training and test list for each one of the attack classification tasks:
         - Attack-type+threat model classification
         - Attack SNR classification
         - Attack threat model classification

   - `run_022_attack_type_classif_allknown.sh`
      - Runs experiment for attack type classification assuming that all attacks types are known in training
      - Trains attack signature network
      - Make T-SNE plots
      - Evaluates attack classification and prints Acc and confusion matrices

   - `run_023_snr_classif_allknown.sh`
      - Runs experiment for attack SNR classification assuming that all attacks types are known in training
      - Trains attack signature network
      - Make T-SNE plots
      - Evaluates attack classification and prints Acc and confusion matrices


   - `run_024_threat_model_classif_allknown.sh`
      - Runs experiment for attack threat model classification assuming that all attacks types are known in training
      - Trains attack signature network
      - Make T-SNE plots
      - Evaluates attack classification and prints Acc and confusion matrices

   - `run_030_make_known_unknown_lists.sh`
      - Makes list for experiments where some attacks are known in training and some are unknown.

   - `run_031_attack_type_verif_and_noveltydet.sh`
      - Runs experiment for attack type verification and novel attack detection when some attacks
        are unknown.
      - Trains signature networks and PLDA only on known attacks
      - Make T-SNE plots
      - Runs Attack verfication trials to decide if 2 recordings contain the same or different attack type
      - Runs Attack novelty detection to decide if the attack in the test sample is known in training or not

   - `run_032_snr_verif.sh`
      - Runs experiment for attack SNR verification and when some attacks types
        are unknown.
      - Trains signature networks and PLDA only on known attacks
      - Make T-SNE plots
      - Runs Attack verfication trials to decide if 2 recordings contain the same or different SNR

   - `run_033_threat_model_verif.sh`
      - Runs experiment for attack threat model verification and when some attacks types
        are unknown.
      - Trains signature networks and PLDA only on known attacks
      - Make T-SNE plots
      - Runs Attack verfication trials to decide if 2 recordings contain the same or different SNR

   - `run_040_extract_xvectors_for_spkverif_task.sh`
      - Extracts x-vectors on benign signals for Voxceleb1 original speaker verification task
      - The same as in recipe adv.v1.1

   - `run_041_eval_benign_verification.sh`
      - Evals benign speaker verification trials and train calibration

   - `run_042_generate_verification_attacks.sh`
      - Generates attacks against speaker verification task

   - `run_043_spkverif_attack_type_classif_allknown.sh`
      - Runs attack classification on the attacks against speaker verification
      - Uses signature network trained on attacks against speaker identification

   - `run_044_spkverif_attack_snr_classif_allknown.sh`
      - Runs attack SNR classification on the attacks against speaker verification
      - Uses signature network trained on attacks against speaker identification

   - `run_045_spkverif_attack_threat_model_classif_allknown.sh`
      - Runs attack threat_model classification on the attacks against speaker verification
      - Uses signature network trained on attacks against speaker identification

   - `run_ds.sh`
      - Runs attack classificaton on ASR DeepSpeech attacks
      - Code to generate the attacks is not included

   - `run_espresso.sh`
      - Runs attack classificaton on ASR Espresso attacks
      - Code to generate the attacks is not included



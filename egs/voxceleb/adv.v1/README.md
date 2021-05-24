# VoxCeleb Adversarial Attacks Version 1

Last update 2021/04/22

Recipe to evaluate Adversarial Attacks to x-Vector Speaker Verification Systems

## Threat Model

Speaker verification pipeline where:
  - Enrollment side is not under attack, x-vectors for enrollment utterances are
    pre-computed and storded on disk
  - Test side is under Adversarial Attacks. 
    The attack adds an inperceptible perturbation to the 
    test waveform to make the system to:
        - Classify target trials as non-targets
        - Classify non-target trials as targets

As attacks happen in waveform domain, test x-vectors cannot be precomputed and
need to be recomputed for each trial.
Also, the speaker verification pipeline needs to be fully differentiable from wave to score,
so the attack algorithm can optimize the perturbation noise.

However, to train the x-vector network, this recipe computes acoustic features and speech augmentations off-line.
Look version adv.v1.1, for a newer recipe which computes features 
and augmentations on the fly.

Two broad types of attacks:
    - White-box: the attacker has access to the x-vector model under attack
    - Transfer based Black-box: the attacker doesn't have access to the x-vector model under attack (black-box model),
       but has access to another x-vector model (white-box). Perturvation is obtained from the white-box model
       and used to attack the black-box model.

Multiple attacks algorithms: FGSM, Iter-FGSM, PGD, Carlini-Wagner.

## Citing

  If you use this recipe, please cite:
```
@inproceedings{Villalba2020,
address = {Shanghai, China},
author = {Villalba, Jes{\'{u}}s and Zhang, Yuekai and Dehak, Najim},
booktitle = {Interspeech 2020},
month = {sep},
title = {{x-Vectors Meet Adversarial Attacks : Benchmarking Adversarial Robustness in Speaker Verification}},
year = {2020}
}
```

## Training Data

   - x-Vector network is trained on Voxceleb2 dev + test with augmentations
     - MUSAN noise
     - RIR reverberation

## Test Data

   - Test data is VoxCeleb 1 Original Clean trial list.
   - We don't use the larger Entire and Hard list because of the high computing cost
     of these experiments. 

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it will use ResNet34 as victim model and Residual E-TDNN as transfer model
   - You can change that modifying the configuration script.
   - For example, to use LResNet34 as transfer model use `config_victim_resnet34_transfer_lresnet.v1.sh` 
     when calling each of the steps as
```bash
run_0*.sh --config-file global_conf/config_victim_resnet34_transfer_lresnet.v1.sh
```

## Recipe Steps:

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
          - VoxCeleb2 train+test
          - VoxCeleb1 Original eval sets

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_003_compute_fbank.sh`
      - Computes log-filter-banks acoustic features for all datasets

   - `run_004_prepare_augment.sh`
      - Prepares Kaldi style data directories for augmented training data with MUSAN noise and RIR reverberation.

   - `run_005_compute_fbank_augment.sh
      - Computes log-filter-banks for augmented datasets

   - `run_010_prepare_victim_xvec_train_data.sh`
      - Prepares features train the victim x-vector model
      - Applies sort-time mean normalization and remove silence frames
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.
      - Creates training and validation lists for x-vector training

   - `run_011_train_victim_xvector.sh`
      - Trains the victim x-vector network

   - `run_012_prepare_transfer_xvec_train_data.sh`
      - Prepares features train the transfer white-box x-vector model
      - If training data for victim and tranfer models is the same, it does nothing

   - `run_013_train_transfer_xvector.sh`
      - Trains the transfer white-box x-vector network

   - `run_030_extract_xvectors_victim_model.sh`
      - Exctracts x-vectors for VoxCeleb1 test set using the victim model

   - `run_031_extract_xvectors_transfer_model.sh`
      - Exctracts x-vectors for VoxCeleb1 test set using the transfer model

   - `run_040_eval_be_victim_model.sh`
      - Eval cosine scoring back-end without attack on victim model x-vectors
      - Trains calibration for the victim model scores
      - Results are left in `exp/scores/$nnet_name/cosine/voxceleb1_o_clean_results`

   - `run_041_eval_be_tranfer_model.sh`
      - Eval cosine scoring back-end without attack on transfer model x-vectors
      - Trains calibration for the tranfer model scores
      - Results are left in `exp/scores/$transfer_nnet_name/cosine/voxceleb1_o_clean_results`
   
   - `run_042_eval_victim_from_wav.sh`
      - Eval cosine scoring back-end without attack on victim model x-vectors 
        from the test wave, computing features and x-vectors on the fly.
      - This script is just to check that we get the same result as in step 40.
      - You don't need to run it.
      - Results are left in `exp/scores/$nnet_name/cosine_from_wav/voxceleb1_o_clean_results`

   - `run_043_eval_whitebox_attacks.sh`
      - Eval white box attacks implemented in Hyperion toolkit: FGSM, Iter-FGSM, PGD, Carlini-Wagner
      - Results are left in `exp/scores/$nnet_name/cosine_${attack_related_label}/voxceleb1_o_clean_results`
      - When using option `--do-analysis true` it calculates curves: SNR vs EER, SNR vs actual DCF, Linf vs EER, Linf vs actual DCF
      - Curves are left in `exp/scores/$nnet_name/cosine_${attack_related_label}_eall/`
      - When using `--save-wav true`, it writes adversarial wavs of succesful attacks to disk
      - Wavs are saves to `exp/scores/$nnet_name/cosine_${attack_related_label}/wav`
 
   - `run_044_eval_transfer_blackbox_attacks.sh`
      - Eval transfer black box attacks implemented in Hyperion toolkit: FGSM, Iter-FGSM, PGD, Carlini-Wagner
      - Results are left in `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_${attack_related_label}/voxceleb1_o_clean_results`
      - When using option `--do-analysis true` it calculates curves: SNR vs EER, SNR vs actual DCF, Linf vs EER, Linf vs actual DCF
      - Curves are left in `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_${attack_related_label}_eall/`
      - When using `--save-wav true`, it writes adversarial wavs of succesful attacks to disk
      - Wavs are saves to `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_${attack_related_label}/wav`

   - `run_045_eval_whitebox_attacks_with_randsmooth_defense.sh`
      - Eval white box attacks with Gaussian randomized smoothing defense.
      - Results are left in `exp/scores/$nnet_name/cosine_${attack_related_label}_randsmooth${smooth_sigma}/voxceleb1_o_clean_results`
 
   - `run_053_eval_art_whitebox_attacks.sh`
      - Eval white box attacks implemented in IBM's Adversarial Robustness Toolkit (ART): FGSM, Iter-FGSM, PGD, Carlini-Wagner
      - Results are left in `exp/scores/$nnet_name/cosine_art_${attack_related_label}/voxceleb1_o_clean_results`
      - When using option `--do-analysis true` it calculates curves: SNR vs EER, SNR vs actual DCF, Linf vs EER, Linf vs actual DCF
      - Curves are left in `exp/scores/$nnet_name/cosine_art_${attack_related_label}_eall/`
      - When using `--save-wav true`, it writes adversarial wavs of succesful attacks to disk
      - Wavs are saves to `exp/scores/$nnet_name/cosine_art_${attack_related_label}/wav`

   - `run_054_eval_art_transfer_blackbox_attacks.sh`
      - Eval transfer black box attacks implemented in IBM's Adversarial Robustness Toolkit (ART): FGSM, Iter-FGSM, PGD, Carlini-Wagner
      - Results are left in `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_art_${attack_related_label}/voxceleb1_o_clean_results`
      - When using option `--do-analysis true` it calculates curves: SNR vs EER, SNR vs actual DCF, Linf vs EER, Linf vs actual DCF
      - Curves are left in `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_art_${attack_related_label}_eall/`
      - When using `--save-wav true`, it writes adversarial wavs of succesful attacks to disk
      - Wavs are saves to `exp/scores/$nnet_name/transfer.$transfer_nnet/cosine_art_${attack_related_label}/wav`

# NIST SRE20 CTS V1

Last update 2021/06/29

Recipe for NIST Speaker Recognition Evaluation2020  CTS condition
Using Hyperion toolkit Pytorch x-vectors and numpy back-ends

## Citing

   If you use this recipe in your work, please cite these related works:

```
@inproceedings{Villalba2020,
address = {Tokyo, Japan},
author = {Villalba, Jes{\'{u}}s and Garcia-Romero, Daniel and Chen, Nanxin and Sell, Gregory and Borgstrom, Jonas and McCree, Alan and Garcia-Perera, Leibny Paola and Kataria, Saurabh and Nidadavolu, Phani Sankar and Torres-Carrasquillo, Pedro A. and Dehak, Najim},
booktitle = {Proceedings of Odyssey 2020- The Speaker and Language Recognition Workshop},
title = {{Advances in Speaker Recognition for Telephone and Audio-Visual Data : the JHU-MIT Submission for NIST SRE19}},
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
url = {https://linkinghub.elsevier.com/retrieve/pii/S0885230819302700},
volume = {60},
year = {2020}
}

@inproceedings{Villalba2019,
address = {Graz, Austria},
author = {Villalba, Jes{\'{u}}s and Chen, Nanxin and Snyder, David and Garcia-Romero, Daniel and McCree, Alan and Sell, Gregory and Borgstrom, Jonas and Richardson, Fred and Shon, Suwon and Grondin, Francois and Dehak, Reda and Garcia-Perera, Leibny Paola and Povey, Daniel and Torres-Carrasquillo, Pedro A. and Khudanpur, Sanjeev and Dehak, Najim},
booktitle = {Proceedings of the 20th Annual Conference of the International Speech Communication Association, INTERSPEECH 2019},
month = {sep},
title = {{State-of-the-art Speaker Recognition for Telephone and Video Speech: the JHU-MIT Submission for NIST SRE18}},
year = {2019}
}

```

## Training Data

   - Switchboard Cellular 1+2
   - Switchboard 2 Phase 1+2+3
   - Fisher Spanish
   - MIXER 6 telephone data
   - NIST SRE04-12 telephone data
   - VoxCeleb1 + VoxCeleb2 dev
   - NIST SRE16 DEV CEB + CMN
   - NIST SRE16 EVAL TGL + YUE 60%
   - NIST SRE18 CMN Dev Unlabeled
   - NIST SRE18 CMN Dev+Eval

## Dev Data

   - NIST SRE16 EVAL TGL + YUE 40%
   - NIST SRE19 CTS Dev, Eval

## Test Data

   - NIST SRE20-CTS

## Usage

   - Run the `run_0*.sh` scripts in sequence
   - By default it will use ResNet34 xvector
   - To use other x-vector models, pass config file as argument to the scripts, e.g.,
```bash
...
run_011_train_xvector.sh --config-file config_fbank64_stmn_resnet34_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.1.sh
run_030_extract_xvectors.sh --config-file config_fbank64_stmn_resnet34_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.1.sh
...
run_042a_eval_be_v3.sh --config-file config_fbank64_stmn_resnet34_arcs30m0.3_adam_lr0.01_amp.v1.ft_w0.1.sh
...
```

## Recipe Steps

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
        all training and test data

   - `run_002_compute_evad.sh`
      - Computes Energy VAD for all datasets

   - `run_003_prepare_noises_rirs.sh`
      - Prepares MUSAN noises, music to be used by SpeechAugment class.
      - Creates Babble noise from MUSAN speech to be used by SpeechAugment class.
      - Prepares RIRs by compacting then into HDF5 files, to be used by SpeechAugment class.

   - `run_010_preproc_audios_for_nnet_train.sh `
      - Transforms all the audios that we are going to use to train the x-vector into a common format, e.g., .flac.
      - Removes silence from the audios
      - Removes utterances shorter than 4secs and speakers with less than 8 utterances.

   - `run_011_combine_xvec_train_data.sh`
      - Combines all datasets that are going to be used to train the x-vector in a single one
      - Creates training and validation lists for x-vector training
      
   - `run_012_train_xvector.sh`
      - Trains the x-vector network on 4sec utts

   - `run_013_finetune_xvector.sh`
      - Fine-tune x-vector network on 10-60 secs utts

   - `run_030_extract_xvectors.sh`
      - Extracts x-vectors for PLDA training and eval using English model
      - By default it uses x-vector network from step 13

   - `run_040a_eval_be_v1.sh, run_041a_eval_be_v2.sh, run_042a_eval_be_v3.sh run_043a_eval_be_v4.sh run_044a_eval_be_knn_v1.sh run_045a_eval_be_knn_v3.sh`
      - Evals 5 different back-ends on the English x-vectors
           - V1: LDA + LNorm + PLDA adapted with SRE18 unlabeled
	   - V2: Cosine Scoring
	   - V3: LDA + LNorm + PLDA adapted with SRE18 labeled
	   - V4: CORAL + LDA + LNorm + PLDA adapted with SRE18 labeled+unlabeled
	   - kNN-V1: Back-end Trained on k-NN speakers
	   - kNN-V3: Back-end Trained on k1-NN speakers, adapted to k2-NN speakers, k2<k1
      - Calibration condition indepedent
      - Results are left in `exp/scores/fbank64_stmn_resnet34_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/*/*cal_v1sre16-yue/*_results`

   - `run_043b_eval_be_v4.sh run_044b_eval_be_knn_v1.sh run_045b_eval_be_knn_v3.sh`
      - Back-end are the same as above
      - Calibration depends on the number of enrollment segments
      - Results are left in `exp/scores/fbank64_stmn_resnet34_eina_hln_chattstatsi128_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1.alllangs_nocv_nocnceleb.ft_10_60_arcm0.3_sgdcos_lr0.05_b128_amp.v2/*/*cal_v2sre16-yue/*_results`	

   - `run_fus*.sh`
      - Fusion examples
      
## Results

TODO


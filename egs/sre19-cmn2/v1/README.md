# NIST SRE19 CallMyNet2 V1

Last update 2020/08/06

Recipe for NIST Speaker Recognition Evaluation CTS condition
using CallMyNet2 data in Tunisian Arabic
Using Kaldi x-vector tools and Hyperion toolkit back-ends

## Training Data

   - MIXER 6 telephone data
   - NIST SRE04-12 telephone data
   - VoxCeleb1 + VoxCeleb2 dev
   - NIST SRE18 CMN Dev Unlabeled
   - NIST SRE18 CMN Dev+Eval 60% of speakers

## Test Data

   - NIST SRE18 CMN Dev+Eval 40% of speakers uses as developement
   - NIST SRE19 CTS Eval

## Usage

   - Run the `run_0*.sh` scripts in sequence

## Recipe Steps

   - `run_001_prepare_data.sh`
      - Data preparation script to generate Kaldi style data directories for 
        all training and test data

   - `run_002_compute_mfcc_evad.sh`
      - Computes MFCC Energy VAD for all datasets

   - `run_003_prepare_augment.sh`
      - Prepares Kaldi style data directories for augmented training data with MUSAN noise and RIR reverberation.

   - `run_004_compute_fbank_augment.sh
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
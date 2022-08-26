# Why this is directory is created: To run DINO-based SSL for utterance-level embedding learning

# How this directory is created:
## 1. Run below in the parent directory:
## cp -r v1.1 dinossl.v1
## 2. STOPGAP: To train a model in the CLSP grid W/O data prep. The data prep. part (before run_011*) will be updated after updating training-related codes first.
## ln -s /export/c01/jcho/hyperion_DINO/egs/voxceleb/v2/dataa data (for dinossl)

# (WIP) training script
## (GOOD) bash run_511_train_xvector.sh --ngpu 1
## (TODO: for multiple gpus)

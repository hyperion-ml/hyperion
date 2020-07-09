# VoxCeleb Version 1

Recipe for the VoxCeleb Speaker Verification Task using several flavors of x-Vectors

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
   - For better performance use full ResNet (64 channels) using `config_resnet34_adam.v2.sh` file as
```bash
run_010_train_resnet_xvector.sh --config-file config_resnet34_adam.v2.sh
run_030_extract_xvectors.sh --config-file config_resnet34_adam.v2.sh
run_030_eval_be.sh --config-file config_resnet34_adam.v2.sh
```

   - To train with mixed precision training use config file `config_resnet34_adam_amp.v2.sh`
     - This runs but I haven't checked whether the final performance is the same as full precison training

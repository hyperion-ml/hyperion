# Dihard 2019 

Recipe for the Dihard 2019 Single Channel condition 

## Training Data

   - x-Vector network is trained on Voxceleb1+2 dev + test with augmentations
     - MUSAN noise
     - RIR reverberation


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

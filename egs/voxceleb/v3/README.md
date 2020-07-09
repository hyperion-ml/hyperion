# VoxCeleb Version 3

Last update 2020/07/09

This recipe is work in progress

Recipe to evaluate generative models on VoxCeleb
We train models on VoxCeleb2 and evaluate on full VoxCeleb1.
The goal is to evaluate the hability of generative models to 
recostruct VoxCeleb1 data or to generate data from scratch.

## Models included:

  The following models can be evaluated with this recipe:
  - Basic Autoencoders (AE)
  - Variational Autoencoders (VAE
  - VQ-VAE
  - Denoising AE, VAE, VQ-VAE

## Training Data

  - Autoencoders, VAE, VQ-VAE, GAN are trained on 
     - VoxCeleb2 dev+test
  - Denoising versions are trained on
     - VoxCeleb2 dev+test + augmentation with 
        - MUSAN noise
        - RIR reverberation

## Test Data

   - Test data is the full VoxCeleb 1

## Usage

   - Run the run_stepnumber_*.sh scripts in sequence
   - Depending on the model that you are testing you can skip some steps
       - if not running denoising versions skip steps 3 and 4
       - Run train/eval steps only corresponding to the model that you are using

## Results

We compute average of the metrics across VoxCeleb1, values in parenthesis are std.

### Models trained without augmentation

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) |
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   |
| config_vae_dc1d_b4d256_z80_c8.opt.lr0.01.v1.sh | VAE | DC1d Enc-Dec <br> DC-blocks=4 dim / Hidden-dim=256 | 80 | 8 | -1.96 (0.62) | 1.57 (0.91) | 0.90 (0.24) |



### Models trained with augmentation (Denoising versions)

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) |
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   |

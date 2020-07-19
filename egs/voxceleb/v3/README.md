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

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) | codebook-size | PPL |
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   | :----: | :----: | 
| config_vae_dc1d_b4d256_z80_c8.opt.lr0.01.v1.sh | VAE | DC1d Enc-Dec <br> dc-blocks=4 / hid-channels=256 | 80 | 8 | -1.96 (0.62) | 1.57 (0.91) | 0.90 (0.24) |
| config_vae_dc1d_b9d256_z80_c8.opt.lr0.01.v1.sh | VAE | DC1d Enc-Dec <br> dc-blocks=9 / hid-channels=256 | 80 | 8 | -1.95 (0.62) | 1.56 (0.91) | 0.89 (0.24) |
| config_vae_resnet1d_b4d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=4/ hid-channels=256 | 80 | 8 | -1.97 (0.65) | 1.55 (0.93) | 0.89 (0.25) |
| config_vae_resnet1d_b8d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=8/ hid-channels=256 | 80 | 8 | -1.98 (0.65) | 1.55 (0.93) | 0.88 (0.25) |
| config_vae_resnet1d_b16d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=16/ hid-channels=256 | 80 | 8 | -1.98 (0.69) | 1.54 (0.94) | 0.88 (0.25) | 
| config_vae_dc2d_b4c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | DC2d Enc-Dec <br> dc-blocks=4 / hid-channels=64 | 80 | 0.8 | -2.25 (1.00) | 1.49 (1.06) | 0.84 (0.29) |
| config_vae_dc2d_b8c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | DC2d Enc-Dec <br> dc-blocks=8 / hid-channels=64 | 80 | 0.8 | -2.23 (1.00) | 1.49 (1.06) | 0.84 (0.29) |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512_c2275.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 2275 | -1.84 (0.21) | 2.20 (0.71) | 1.12 (0.16) | 512 | 71 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x2_c1138.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 1138  | -1.79 (0.32) | 1.86 (0.78) | 1.01 (0.19) | 512x2 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x4_c569.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 569  | -1.40 (0.43) | 1.69 (0.83) | 0.95 (0.21) | 512x4 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x4_c569_predvar.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 569  | -1.78 (0.42) | 1.70 (0.83) | 0.95 (0.21) | 512x4 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x8_c284.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 284  | -1.87 (0.59) | 1.56 (0.89) | 0.89 (0.23) | 512x8 | 65 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x16_c142.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 142  | -2.04 (0.83) | 1.46 (0.96) | 0.84 (0.27) | 512x16 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x32_c71.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 71  | -2.15 (1.4) | 1.43 (1.08) | 0.80 (0.32) | 512x32 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x64_c36.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 36  | -9.27 (8.31) | 1.49 (1.22) | 0.79 (0.36) | 512x64 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x128_c18.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 18  | -20.97 (20.62) | 1.46 (1.24) | 0.77 (0.38) | 512x128 | 73 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x256_c9.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 9  | -27.91 (26.00) | 1.49 (1.27) | 0.78 (0.39) | 512x256 | 67 |

### Models trained with augmentation (Denoising versions)

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) |
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   |
| config_dvae_resnet1d_b16d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=16 / hid-channels=256 | 80 | 8 | -1.77 (0.33) | 1.67 (0.87) | 0.94 (0.22) |
| config_dvae_resnet2d_b16c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | ResNet2d Enc-Dec <br> res-blocks=16 / base-channels=64 | 80 | 0.8 | -1.77 (0.39) | 1.57 (0.92) | 0.89 (0.25) |

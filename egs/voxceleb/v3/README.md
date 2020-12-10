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
We report EER on VoxCeleb1 Test Original Clean Task using recostructed log-filter-banks and LResNet34 x-vector trained in recipe v1.1.
Baseline EER=1.94% when using original log-filter-banks.

### Models trained without augmentation

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) | codebook size | EER(%) | 
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   | :----: | :----: | 
| config_vae_dc1d_b4d256_z80_c8.opt.lr0.01.v1.sh | VAE | DC1d Enc-Dec <br> dc-blocks=4 / hid-channels=256 | 80 | 8 | -1.96 (0.62) | 1.57 (0.91) | 0.90 (0.24) | | 16.36 |
| config_vae_dc1d_b9d256_z80_c8.opt.lr0.01.v1.sh | VAE | DC1d Enc-Dec <br> dc-blocks=9 / hid-channels=256 | 80 | 8 | -1.95 (0.62) | 1.56 (0.91) | 0.89 (0.24) |
| config_vae_resnet1d_b4d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=4/ hid-channels=256 | 80 | 8 | -1.97 (0.65) | 1.55 (0.93) | 0.89 (0.25) | | 15.05 |
| config_vae_resnet1d_b8d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=8/ hid-channels=256 | 80 | 8 | -1.98 (0.65) | 1.55 (0.93) | 0.88 (0.25) | | 13.45 |
| config_vae_resnet1d_b16d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=16/ hid-channels=256 | 80 | 8 | -1.98 (0.69) | 1.54 (0.94) | 0.88 (0.25) | | 13.45 |
| config_vae_dc2d_b4c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | DC2d Enc-Dec <br> dc-blocks=4 / hid-channels=64 | 80 | 0.8 | -2.25 (1.00) | 1.49 (1.06) | 0.84 (0.29) | | 10.04 |
| config_vae_dc2d_b8c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | DC2d Enc-Dec <br> dc-blocks=8 / hid-channels=64 | 80 | 0.8 | -2.23 (1.00) | 1.49 (1.06) | 0.84 (0.29) |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512_c2275.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 2275 | -1.84 (0.21) | 2.20 (0.71) | 1.12 (0.16) | 512 |  28.42 | 
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x2_c1138.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 1138  | -1.79 (0.32) | 1.86 (0.78) | 1.01 (0.19) | 512x2 | 22.08 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x4_c569.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 569  | -1.40 (0.43) | 1.69 (0.83) | 0.95 (0.21) | 512x4 | 19.18 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x4_c569_predvar.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 569  | -1.78 (0.42) | 1.70 (0.83) | 0.95 (0.21) | 512x4 | 18.16 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x8_c284.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 284  | -1.87 (0.59) | 1.56 (0.89) | 0.89 (0.23) | 512x8 | 15.48 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x16_c142.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 142  | -2.04 (0.83) | 1.46 (0.96) | 0.84 (0.27) | 512x16 | 11.77 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x32_c71.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 71  | -2.15 (1.4) | 1.43 (1.08) | 0.80 (0.32) | 512x32 | 8.13 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x64_c36.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 36  | -9.27 (8.31) | 1.49 (1.22) | 0.79 (0.36) | 512x64 | 6.41 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x128_c18.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 18  | -20.97 (20.62) | 1.46 (1.24) | 0.77 (0.38) | 512x128 | 5.67 |
| config_vqvae_resnet1d_b8d256_emakmeansvq_z256cb512x256_c9.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 9  | -27.91 (26.00) | 1.49 (1.27) | 0.78 (0.39) | 512x256 | 5.41 |
| config_vqvae_transformer_b6d512h8ff2048_emakmeansvq_z512cb512x8_c36.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6/ d_model=512 / heads=8 / d_ff=2048 | 512 | 36 |  -1.74(0.31) | 0.48 (0.15) | 0.52 (0.08) | 512x8 | 10.49 |
| config_vqvae_transformer_lac25b6d512h8ff2048_emakmeansvq_z512cb512x8_c36.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6 / d_model=512 / heads=8 / att-context=25 / d_ff=2048 | 512 | 36 |  -1.61(0.15) | 0.42 (0.08) | 0.49 (0.05) | 512x8 | 4.26 |
| config_vqvae_transformer_lac25b6d512h8ff2048_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6 / d_model=512 / heads=8 / att-context=25 / d_ff=2048 <br> RAdam Opt. | 512 | 36 |  -1.33(0.15) | 0.28 (0.05) | 0.40 (0.03) | 512x8 | 4.06 | 
| config_vqvae_transformer_b6d512h8ff2048rpe_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6 / d_model=512 / heads=8 / d_ff=2048 <br> Rel. Pos Enc. <br> RAdam Opt. | 512 | 36 |  -1.29(0.10) | 0.27 (0.05) | 0.39 (0.03) | 512x8 | 4.21 |
| config_vqvae_transformer_lac25b6d512h8ff2048rpe_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6 / d_model=512 / heads=8 / att-context=25 / d_ff=2048 <br> Rel. Pos Enc. <br> RAdam Opt. | 512 | 36 |  -1.30(0.09) | 0.27 (0.04) | 0.39 (0.03) | 512x8 | 4.02 |
| config_vqvae_conformer_lac25b6d512h8cbk31ff2048_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.01.v4.sh | VQ-VAE | Conformer Enc <br> blocks=6 / d_model=512 / heads=8 / att-context=25 / d_ff=2048 <br> RAdam Opt. | 512 | 36 |  -1.26(0.10) | 0.28 (0.04) | 0.39 (0.03) | 512x8 | 4.06 |


### Models trained with augmentation (Denoising versions)

| Config | Model Type | Architecture |  Latent-channels | Compression (bits x/bits z) | ELBO/dim (std) | MSE (std) | L1 (std) | codebook size | EER(%) | 
| ------ | ---------- | ------------ | :--------: | :-------: | :----:   | :----:   | :----:   | :----:   | :----:   | 
| config_dvae_resnet1d_b16d256_z80_c8.opt.lr0.01.v1.sh | VAE | ResNet1d Enc-Dec <br> res-blocks=16 / hid-channels=256 | 80 | 8 | -1.77 (0.33) | 1.67 (0.87) | 0.94 (0.22) | | 16.70 |
| config_dvae_resnet2d_b16c64_z80_c0.8.opt.lr0.01.v1.sh | VAE | ResNet2d Enc-Dec <br> res-blocks=16 / base-channels=64 | 80 | 0.8 | -1.77 (0.39) | 1.57 (0.92) | 0.89 (0.25) | | 12.40 |
| config_vqdvae_resnet1d_b8d256_emakmeansvq_z256cb512x4_c569.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 569  | -1.75 (0.29) | 1.78 (0.84) | 0.98 (0.21) | 512x4 | 18.37 |
| config_vqdvae_resnet1d_b8d256_emakmeansvq_z256cb512x8_c284.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 284  | -1.80 (0.42) | 1.69 (0.83) | 0.95 (0.21) | 512x8 | 15.19 |
| config_vqdvae_resnet1d_b8d256_emakmeansvq_z256cb512x16_c142.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 142  | -1.81 (0.42) | 1.55 (0.97) | 0.87 (0.26) | 512x16 | 11.37 |
| config_vqdvae_resnet1d_b8d256_emakmeansvq_z256cb512x32_c71.opt.lr0.01.v1.sh | VQ-VAE | ResNet1d Enc-Dec <br> res-blocks=8 / hid-channels=256 | 256 | 71  | -1.95 (0.49) | 1.47 (1.03) | 0.83 (0.30) | 512x32 | 8.75 |
| config_vqdvae_transformer_lac25b6d512h8ff2048_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.01.v4.sh | VQ-VAE | Xformer Enc <br> blocks=6/ d_model=512 / heads=8 / att-context=25 / d_ff=2048 <br> Radam Opt. | 512 | 36 | -1.85 (0.13) | 0.56 (0.31) | 0.57 (0.11) | 512x8 | 5.3 |
| config_vqdvae_transformer_lac25b6d512h8ff2048rpe_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.005.v6.sh | VQ-VAE | Xformer Enc <br> blocks=6/ d_model=512 / heads=8 / d_ff=2048 <br> Rel. Pos. Enc <br> Radam Opt. | 512 | 36 | -1.77 (0.05) | 0.43 (0.10) | 0.51 (0.04) | 512x8 | 4.56 |
| config_vqdvae_conformer_lac25b6d512h8cbk31ff2048_emakmeansvq_z512cb512x8_c36_radam.opt.lr0.0025.v6.sh | VQ-VAE | Conformer Enc <br> blocks=6/ d_model=512 / heads=8 / d_ff=2048 <br> Rel. Pos. Enc <br> Radam Opt. | 512 | 36 | -1.83 (0.05) | 0.59 (0.11) | 0.59 (0.04) | 512x8 | 6.56 |



# SRE19-AV-V/v0.2

Recipe for face recognition in videos based on PyTorch pre-trained RetinaFace face detector and ArcFace Embeddings

## Dependencies

```
conda create --name hyperion_tyche python=3.8
conda activate hyperion_tyche
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

We use PyAV package to read the videos
```
https://docs.mikeboers.com/pyav/develop/index.html
```
Install in conda with:
```bash
conda install av -c conda-forge
```
it requeres ffmepg available in the grid.

Install opencv and other requirements
```
pip install -r requirements.txt
```

Get pytorch-insightface repo
```
local/install_foamliu_insightface.sh
```

## Citing

This recipe intends to replicate JHU-HLTCOE team's NIST SRE19 submission, cite:
```
@inproceedings{Villalba2020,
address = {Tokyo, Japan},
author = {Villalba, Jes{\'{u}}s and Garcia-Romero, Daniel and Chen, Nanxin and Sell, Gregory and Borgstrom, Jonas and McCree, Alan and {Garcia Perera}, Leibny Paola and Kataria, Saurabh and Nidadavolu, Phani Sankar and Torres-Carrasquillo, Pedro and Dehak, Najim},
booktitle = {Odyssey 2020 The Speaker and Language Recognition Workshop},
doi = {10.21437/Odyssey.2020-39},
month = {nov},
pages = {273--280},
publisher = {ISCA},
title = {{Advances in Speaker Recognition for Telephone and Audio-Visual Data: the JHU-MIT Submission for NIST SRE19}},
url = {http://www.isca-speech.org/archive/Odyssey_2020/abstracts/88.html},
year = {2020}
}
```

You may want to cite also:

```
@inproceedings{yang2016wider,
title = {WIDER FACE: A Face Detection Benchmark},
author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
booktitle = {CVPR},
year = {2016}
}
  
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```


## Training Data
   We use pre-trained models from InsightFace-Pytorch repo.
   - RetinaNet trained on Widerface
   - ArcFace face embeddings trained on MS-Celeb-1M

## Test data

   We evaluate:
     - SRE19 AV dev/eval
     - Janus Core dev/eval

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it uses LResNet100 ArcFace
   - To choose other network use config files as
```bash
run_0xx_....sh --config-file global_conf/config_r50.sh
```

## Recipe Steps

   - `run_000_get_pretrained_models.sh`
     - Download pre-trained models, if links don't work download manually from the webpages and put then in
     `./exp/face_models`
     
   - `run_001_prepare_data.sh`
     - Data preparation script to generate Kaldi style data directories for 
       - SRE19 AV/JANUS

   - `run_002_extract_embed.sh`
     - Detect faces and computes face embeddings
     
   - `run_040_eval_be_v1.sh`
     - Evaluates backend v1
       - Score all enroll face-embeddings versus all test face-embeddings and take the max. score.
     
   - `run_041_eval_be_v2.sh`
     - Evaluates backend v2
       - Score average enroll face-embeddings versus all test face-embeddings and take the max. score.
     
   - `run_042_eval_be_v3.sh`
     - Evaluates backend v3
       - Score median enroll face-embeddings versus all test face-embeddings and take the max. score.
     
   - `run_043_eval_be_v4.sh`
     - Evaluates backend v4
       - Enroll on median face-embeddings
       - Test performs agglomerative clustering of face-embeddings
       - Score enrollment embedding versus mean of each cluster and take the maximum score
     
   - `run_044_eval_be_v5.sh`
     - Evaluates backend v5
       - Enroll performs agglomerative clustering of face-embeddings
       - Test performs agglomerative clustering of face-embeddings
       - Score all enrollment clusters versus all test clusters and take the maximum score.

   - `run_045_eval_be_v6.sh`
     - Evaluates backend v6
       - Enroll on median face-embeddings
       - Test performs refinement of embddings with self-attention mechanism
       - Score enrollment embedding versus refined embeddings take the maximum score

   - `run_046_eval_be_v7.sh`
     - Evaluates backend v7
       - Enroll on median face-embeddings
       - Test source-target attention mechanism between enrollment embedding and test embeddings to obtain a single test embedding close to the enrollment.
       - Score enrollment embedding versus test embedding

     
   - `run_048_eval_be_v9.sh`
     - Evaluates backend v9
       - Enroll performs refinement of embddings with self-attention mechanism
       - Test performs refinement of embddings with self-attention mechanism
       - Score all refined enrollment embeddings versus all test refined embeddings take the maximum score

  Description of the back-ends is in the JHU-MIT SRE19 paper cited above.
  
| Config | Model Type | Back-end | SRE19 DEV AV |     |     | SRE19 EVAL AV |     |     | JANUS DEV CORE |     |     | JANUS EVAL CORE |     |     |
| ------ | ---------- | -------- | :----------: | :-: | :-: | :-----------: | :-: | :-: | :------------: | :-: | :-: | :-------------: | :-: | :-: |
|        |            |          | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp | EER | Min Cp | Act Cp |
| default_config.sh | LResNet100 | v1 | 7.32 | 0.432 | 0.473 | 2.71 | 0.306 | 0.376 | 3.65 | 0.127 | 0.478 | 1.30 | 0.097 | 0.356 |
| | | v1 + s-norm |  8.88 | 0.207 | 0.211 | 1.98 | 0.063 | 0.065 | 5.42 | 0.118 | 0.295 | 1.36 | 0.054 | 0.149 |
| | | v2 | 5.28 | 0.440 | 0.462 | 1.55 | 0.055 | 0.155 | 3.92 | 0.117 | 0.408 | 1.35 | 0.070 | 0.277 | 
| | | v2 + s-norm | | 8.98 | 0.192 | 0.208 | 1.99 | 0.061 | 0.067 | 4.44 | 0.122 | 0.284 | 1.67 | 0.053 | 0.138 |
| | | v3 | 4.98 | 0.444 | 0.468 | 1.61 | 0.059 | 0.162 | 3.96 | 0.118 | 0.384 | 1.49 | 0.072 | 0.262 | 
| | | v3 + s-norm |  8.89 | 0.198 | 0.208 | 1.95 | 0.067 | 0.070 | 4.44 | 0.121 | 0.272 | 1.66 | 0.058 | 0.137 |
| | | v4 | 4.84 | 0.454 | 0.477 | 1.55 | 0.054 | 0.164 | 3.95 | 0.117 | 0.388 | 1.40 | 0.070 | 0.258 |
| | | v4 + s-norm | 8.58 | 0.194 | 0.205 | 1.89 | 0.066 | 0.070 | 4.30 | 0.124 | 0.268 | 1.62 | 0.054 | 0.132 |
| | | v5 | 7.76 | 0.456 | 0.477 | 2.67 | 0.315 | 0.385 | 3.74 | 0.130 | 0.508 | 1.34 | 0.104 | 0.397 |
| | | v5 + s-norm |  8.58 | 0.199 | 0.211 | 2.08 | 0.059 | 0.061 | 4.80 | 0.128 | 0.296 | 1.31 | 0.051 | 0.157 |
| | | v6 | 5.34 | 0.216 | 0.230 | 2.59 | 0.065 | 0.075 | 4.25 | 0.094 | 0.194 | 1.26 | 0.056 | 0.119 |
| | | v6 + s-norm | 7.54 | 0.194 | 0.201 | 2.80 | 0.078 | 0.083 | 3.79 | 0.114 | 0.221 | 1.56 | 0.048 | 0.103 |
| | | v7 |  5.50 | 0.207 | 0.235 | 4.17 | 0.086 | 0.097 | 5.01 | 0.108 | 0.194 | 2.02 | 0.081 | 0.132 |
| | | v7 + s-norm | 9.06 | 0.223 | 0.227 | 4.31 | 0.097 | 0.100 | 4.87 | 0.124 | 0.254 | 1.99 | 0.083 | 0.131 |
| | | v9 |  5.22 | 0.206 | 0.208 | 2.26 | 0.067 | 0.083 | 4.18 | 0.090 | 0.225 | 1.11 | 0.058 | 0.155 |
| | | v9 + s-norm |  6.37 | 0.186 | 0.196 | 2.78 | 0.073 | 0.078 | 3.72 | 0.111 | 0.238 | 1.31 | 0.049 | 0.103 | 

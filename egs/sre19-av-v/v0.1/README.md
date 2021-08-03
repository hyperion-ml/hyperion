# SRE19-AV-V/v0.1

Recipe for face recognition in videos based on insightface repo

## Dependencies

This toolkit requires MX-Net, we create a separate environment for mxnet,
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

Install mxnet, opencv and other requirements
```
pip install -r requirements.txt
```

Get insightface repo
```
local/install_insightface.sh
```

## Citing

We used this setup in JHU-CLSP team submission to NIST SRE19, cite:
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
   We use pre-trained models from InsightFace repo.
   - RetinaNet trained on Widerface
   - ArcFace face embeddings trained on MS1M

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
| default_config.sh | ResNet100 | v1 | 11.42 | 0.324 | 0.324 | 4.30 | 0.147 | 0.179 | 4.32 | 0.130 | 0.375 | 3.76 | 0.103 | 0.274 | 
| | | v1 + s-norm | 11.86 | 0.224 | 0.242 | 3.59 | 0.086 | 0.092 | 5.63 | 0.119 | 0.320 | 3.78 | 0.081 | 0.185 | 
| | | v2 | 8.16 | 0.352 | 0.391 | 3.72 | 0.075 | 0.106 | 4.67 | 0.141 | 0.334 | 3.44 | 0.091 | 0.229 | 
| | | v2 + s-norm | 11.26 | 0.231 | 0.249 | 3.50 | 0.086 | 0.089 | 4.87 | 0.120 | 0.284 | 3.68 | 0.080 | 0.144 | 
| | | v3 | 8.05 | 0.330 | 0.377 | 3.67 | 0.077 | 0.107 | 5.30 | 0.145 | 0.322 | 3.45 | 0.092 | 0.223 | 
| | | v3 + s-norm | 11.67 | 0.231 | 0.286 | 3.56 | 0.081 | 0.083 | 5.62 | 0.129 | 0.296 | 3.67 | 0.079 | 0.153 | 
| | | v4 | 7.98 | 0.346 | 0.380 | 3.63 | 0.073 | 0.098 | 5.23 | 0.146 | 0.326 | 3.49 | 0.092 | 0.221 | 
| | | v4 + s-norm | 11.39 | 0.224 | 0.277 | 3.51 | 0.079 | 0.081 | 5.61 | 0.130 | 0.292 | 3.59 | 0.075 | 0.158 | 
| | | v5 | 11.37 | 0.354 | 0.354 | 4.30 | 0.158 | 0.186 | 4.23 | 0.121 | 0.370 | 3.62 | 0.110 | 0.296 | 
| | | v5 + s-norm | 11.19 | 0.218 | 0.236 | 3.54 | 0.079 | 0.082 | 5.23 | 0.123 | 0.324 | 3.49 | 0.079 | 0.193 | 
| | | v6 | 7.89 | 0.295 | 0.306 | 4.66 | 0.096 | 0.108 | 5.66 | 0.131 | 0.259 | 3.35 | 0.085 | 0.183 | 
| | | v6 + s-norm | 10.15 | 0.218 | 0.227 | 4.70 | 0.095 | 0.101 | 5.54 | 0.124 | 0.234 | 3.80 | 0.081 | 0.103 | 
| | | v7 | 8.05 | 0.281 | 0.303 | 6.32 | 0.119 | 0.135 | 6.26 | 0.155 | 0.262 | 4.95 | 0.119 | 0.172 | 
| | | v7 + s-norm | 10.98 | 0.238 | 0.245 | 6.52 | 0.135 | 0.136 | 6.35 | 0.163 | 0.283 | 5.26 | 0.110 | 0.131 | 
| | | v9 | 7.34 | 0.302 | 0.327 | 4.55 | 0.107 | 0.115 | 5.32 | 0.115 | 0.287 | 3.34 | 0.086 | 0.216 | 
| | | v9 + s-norm | 9.52 | 0.215 | 0.236 | 4.76 | 0.096 | 0.097 | 5.13 | 0.115 | 0.246 | 3.78 | 0.078 | 0.107 | 

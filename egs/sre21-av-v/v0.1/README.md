# SRE21-AV-V/v0.1

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
     - SRE21 AV dev/eval
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
       - SRE21 AV/JANUS

   - `run_002_extract_embed.sh`
     - Detect faces and computes face embeddings
     
   - `run_042_eval_be_v3.sh`
     - Evaluates backend v3
       - Score median enroll face-embeddings versus all test face-embeddings and take the max. score.
     
   - `run_043_eval_be_v4.sh`
     - Evaluates backend v4
       - Enroll on median face-embeddings
       - Test performs agglomerative clustering of face-embeddings
       - Score enrollment embedding versus mean of each cluster and take the maximum score
     
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


  Description of the back-ends is in the JHU-MIT SRE19 paper cited above.


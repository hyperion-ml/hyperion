#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1

config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

# If links don't work anymore, go to the webpages and download manually

# if [ $stage -le 1 ];then
#   echo "Download facedet models from https://github.com/foamliu/InsightFace-PyTorch/tree/master/retinaface/weights"
#   mkdir -p $face_det_modeldir
#   cd $face_det_modeldir
#   wget https://github.com/foamliu/InsightFace-PyTorch/blob/master/retinaface/weights/mobilenet0.25_Final.pth
#   cd -
# fi


echo "Download face recognition models from https://github.com/foamliu/InsightFace-PyTorch"
if [ $stage -le 2 ];then
    mkdir -p $face_reco_modeldir
    cd $face_reco_modeldir
    wget https://github.com/foamliu/InsightFace-v3/releases/download/v1.0/insight-face-v3.pt
    cd -
fi

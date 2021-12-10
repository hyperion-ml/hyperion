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

if [ $stage -le 1 ];then
  echo "Download facedet models from https://github.com/deepinsight/insightface/tree/master/detection/retinaface"
  mkdir -p $face_det_modeldir
  cd $face_det_modeldir
  #wget https://www.dropbox.com/s/53ftnlarhyrpkg2/retinaface-R50.zip
  gdown https://drive.google.com/uc?id=1yyaj--_zrBZvfH9ttZWtc40nEyT_P81G
  unzip retinaface-R50.zip
  cd -
fi


echo "Download face recognition models from https://github.com/deepinsight/insightface/wiki/Model-Zoo"
if [ $stage -le 2 ];then

    face_reco_modeldir=exp/face_models/r100-arcface
    mkdir -p $face_reco_modeldir
    cd $face_reco_modeldir
    # wget https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip
    gdown https://drive.google.com/uc?id=1Z7RLxn91ox-WUR6WB_Qr4cFpwvHcnza5
    unzip model-r100-arcface-ms1m-refine-v2.zip
    mv model-r100-ii/* .
    rmdir model-r100-ii
    cd -
fi
exit

if [ $stage -le 3 ];then
    face_reco_modeldir=exp/face_models/r50-arcface
    mkdir -p $face_reco_modeldir
    cd $face_reco_modeldir
    wget https://www.dropbox.com/s/ou8v3c307vyzawc/model-r50-arcface-ms1m-refine-v1.zip
    unzip model-r50-arcface-ms1m-refine-v1.zip
    mv model-r50-am-lfw/* .
    rmdir model-r50-am-lfw
    cd -
fi


if [ $stage -le 4 ];then
    face_reco_modeldir=exp/face_models/r34-arcface
    face_reco_model=$face_reco_modeldir/model
    mkdir -p $face_reco_modeldir
    cd $face_reco_modeldir
    wget https://www.dropbox.com/s/yp7y6jic464l09z/model-r34-arcface-ms1m-refine-v1.zip
    unzip model-r34-arcface-ms1m-refine-v1.zip
    mv model-r34-amf/* .
    rmdir model-r34-amf
    cd -
fi



if [ $stage -le 5 ];then
    face_reco_modeldir=exp/face_models/mobile-arcface
    face_reco_model=$face_reco_modeldir/model
    mkdir -p $face_reco_modeldir
    cd $face_reco_modeldir
    wget https://www.dropbox.com/s/akxeqp99jvsd6z7/model-MobileFaceNet-arcface-ms1m-refine-v1.zip
    unzip model-MobileFaceNet-arcface-ms1m-refine-v1.zip
    mv model-y1-test2/* .
    rmdir model-y1-test2
    cd -
fi


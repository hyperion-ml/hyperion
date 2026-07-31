#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

config_file=default_config.sh
stage=1

. parse_options.sh || exit 1;
. datapath.sh


if [ $stage -le 1 ];then
  # Prepares voxlingua 107 for training
  hyp_utils/conda_env.sh \
    local/prepare_voxlingua107.py \
    --corpus-dir $voxlingua_root \
    --output-dir data/voxlingua107 \
    --remove-langs en-en es-es ar-ar pt-pt \
    --map-langs-to-lre-codes \
    --target-fs 8000
  
fi

if [ $stage -le 2 ];then
  # Prepare LRE17 Training data
  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_train \
    --subset train \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_dev_cts \
    --subset dev \
    --source mls14 \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_train_root \
    --output-dir data/lre17_dev_afv \
    --subset dev \
    --source vast \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_eval_root \
    --output-dir data/lre17_eval_cts \
    --subset eval \
    --source mls14 \
    --target-fs 8000

  hyp_utils/conda_env.sh \
    local/prepare_lre17.py \
    --corpus-dir $lre17_eval_root \
    --output-dir data/lre17_eval_afv \
    --subset eval \
    --source vast \
    --target-fs 8000

fi

if [ $stage -le 3 ];then
  hyp_utils/conda_env.sh \
    local/prepare_lre22_dev.py \
    --corpus-dir $lre22_dev_root \
    --output-dir data/lre22_dev \
    --target-fs 8000

fi

if [ $stage -le 4 ];then
  hyp_utils/conda_env.sh \
    local/prepare_lre22_eval.py \
    --corpus-dir $lre22_eval_root \
    --output-dir data/lre22_eval \
    --target-fs 8000

fi

if [ $stage -le 5 ];then
    local/make_sre16_train_dev.sh $sre16_dev_root 8 data
    local/make_sre16_train_eval.sh $sre16_eval_root 8 data
fi

if [ $stage -le 6 ];then
    local/make_sre18_dev_unlabeled.sh $sre18_dev_root 8 data
    local/make_sre18_train_dev.sh $sre18_dev_root 8 data
    local/make_sre18_train_eval.sh $sre18_eval_root 8 data
fi

if [ $stage -le 7 ];then
    # Prepare sre19
    local/make_sre19cmn2_eval.sh $sre19cmn2_eval_root 8 data
fi

if [ $stage -le 8 ];then
    # Prepare SRE21 dev
    hyp_utils/conda_env.sh \
    	local/prepare_sre21av_dev_audio.py \
    	--corpus-dir $sre21_dev_root \
    	--target-fs 8000 \
    	--output-path data/sre21_audio_dev \
    	--av-output-path data/sre21_audio-visual_dev
    # Prepare SRE21 eval
    hyp_utils/conda_env.sh \
	local/prepare_sre21av_eval_audio.py \
	--corpus-dir $sre21_eval_root \
	--target-fs 8000 \
	--output-path data/sre21_audio_eval \
	--av-output-path data/sre21_audio-visual_eval

fi

if [ $stage -le 9 ];then
    # Prepare SRE CTS superset
    hyp_utils/conda_env.sh \
	local/prepare_sre_cts_superset.py \
	--corpus-dir $sre_superset_root \
	--target-fs 8000 \
        --output-dir data/sre_cts_superset
fi

if [ $stage -le 10 ];then
    # Prepare babel datasets
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_assamese_root \
	--target-fs 8000 \
	--lang-code as-as \
        --output-dir data/babel_assamese
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_bengali_root \
	--target-fs 8000 \
	--lang-code bn-bn \
        --output-dir data/babel_bengali
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_pashto_root \
	--target-fs 8000 \
	--lang-code ps-ps \
        --output-dir data/babel_pashto
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_turkish_root \
	--target-fs 8000 \
	--lang-code tr-tr \
        --output-dir data/babel_turkish
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_georgian_root \
	--target-fs 8000 \
	--lang-code ka-ka \
        --output-dir data/babel_georgian
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_vietnam_root \
	--target-fs 8000 \
	--lang-code vi-vi \
        --output-dir data/babel_vietnam
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_haitian_root \
	--target-fs 8000 \
	--lang-code ht-ht \
        --output-dir data/babel_haitian
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_lao_root \
	--target-fs 8000 \
	--lang-code lo-lo \
        --output-dir data/babel_lao
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_tamil_root \
	--target-fs 8000 \
	--lang-code ta-ta \
        --output-dir data/babel_tamil
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_zulu_root \
	--target-fs 8000 \
	--lang-code zul-zul \
        --output-dir data/babel_zulu
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_kurmanji_root \
	--target-fs 8000 \
	--lang-code kur-kur \
        --output-dir data/babel_kurmanji
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_tok_root \
	--target-fs 8000 \
	--lang-code tok-tok \
        --output-dir data/babel_tok
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_kazakh_root \
	--target-fs 8000 \
	--lang-code kk-kk \
        --output-dir data/babel_kazakh
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_telugu_root \
	--target-fs 8000 \
	--lang-code te-te \
        --output-dir data/babel_telugu
    hyp_utils/conda_env.sh \
	local/prepare_babel.py \
	--corpus-dir $babel_lithuanian_root \
	--target-fs 8000 \
	--lang-code lt-lt \
        --output-dir data/babel_lithuanian

fi

if [ $stage -le 11 ];then
    hyp_utils/conda_env.sh \
	local/prepare_some_data_for_lre.py \
	--corpus-dir $fleurs_root \
	--output-dir data/fleurs22 \
	--map-langs-to-lre-codes --target-fs 8000
    
    hyp_utils/conda_env.sh \
	local/prepare_some_data_for_lre_cat.py \
	--corpus-dir $lwazi_root \
	--output-dir data/lwazi09 \
	--map-langs-to-lre-codes --target-fs 8000
    hyp_utils/conda_env.sh \
	local/prepare_some_data_for_lre_cat.py \
	--corpus-dir $nchlt_root \
	--output-dir data/nchlt14 \
	--map-langs-to-lre-codes --target-fs 8000
    hyp_utils/conda_env.sh \
	local/prepare_some_data_for_lre_cat.py \
	--corpus-dir $ammi_root \
	--output-dir data/ammi20 \
	--map-langs-to-lre-codes --target-fs 8000
fi

if [ $stage -le 12 ];then

    hyp_utils/conda_env.sh \
	local/prepare_common_voice_cat.py \
	--corpus-dir $cv22_root \
	--output-dir data/cv22_tir \
	--keep-langs tir-tir \
	--map-langs-to-lre-codes --target-fs 8000
fi


if [ $stage -le 13 ];then
  hyp_utils/conda_env.sh \
	local/prepare_common_voice_accents_cat.py \
	--corpus-dir $cv20_root \
	--output-dir data/cv20_eng_ine \
	--lang en \
	--target-fs 8000
  hyp_utils/conda_env.sh \
	local/prepare_common_voice_accents_cat.py \
	--corpus-dir $cv20_root \
	--output-dir data/cv20_fra \
	--lang fr \
	--target-fs 8000

fi

if [ $stage -le 14 ];then
      hyp_utils/conda_env.sh \
	  local/prepare_adi17.py \
	  --corpus-dir $adi_root \
	  --output-dir data/adi17 \
	  --map-langs-to-lre-codes --target-fs 8000
fi

if [ $stage -le 15 ];then
    hyp_utils/conda_env.sh \
	local/prepare_ast_cat.py \
	--corpus-dir $ast_root \
	--output-dir data/ast \
	--map-langs-to-lre-codes --target-fs 8000
fi

if [ $stage -le 16 ];then
    #combine data
    utils/combine_data.sh \
	data/babel \
	data/babel_{a*,b*,g*,k*,l*,p*,t*,v*,zulu}

    utils/combine_data.sh \
	data/cv \
	data/cv20_eng_ine data/cv20_fra data/cv22_tir

    utils/combine_data.sh \
	data/sre16 \
	data/sre16_train_{dev*,eval*} 

    utils/combine_data.sh \
	data/sre18 \
	data/sre18_train_{dev*,eval*} data/sre18_dev_unlabeled

    utils/combine_data.sh \
	data/sre19 \
	data/sre19_eval_{enroll,test}_cmn2

    utils/combine_data.sh \
	data/sre21_cts \
	data/sre21_*_cts

    utils/combine_data.sh \
	data/sre21_afv \
	data/sre21_audio*_{dev*,eval*}_afv

    utils/combine_data.sh \
	data/sre16-21_cts \
	data/sre1{6,8,9} data/sre21_cts
    
fi
  
if [ $stage -le 5 ];then
    if [ -d ../fixed.v1.8k/lre-scorer ];then
	ln -s ../fixed.v1.8k/lre-scorer
    else
	local/download_lre22_scorer.sh
    fi
    if [ -d ../fixed.v1.8k/focal_multiclass ];then
	ln -s ../fixed.v1.8k/focal_multiclass
    else
	local/download_focal.sh
    fi
fi

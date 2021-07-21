#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=12800
state_dict_key=model_teacher_state_dict # which state dict to use from pre-trained model between student (model_state_dict) and teacher (model_teacher_state_dict for dinossl). This can be overwritten by ". $config_file" below
dinossl_cfg="" # configuration file that includes 1) the info to reconstruct the DINOHead used in front-end training, 2) location to extract xvectors
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd --mem 4G"
else
    xvec_cmd="$train_cmd --mem 12G"
fi

xvector_dir=exp/xvectors/$nnet_name
chkpnt_idx=`basename ${nnet%.pth}`

if [ "${state_dict_key}" != "model_state_dict" ];then
    echo "Extracting xvectors from teacher model"
    xvec_args+=" --state_dict_key ${state_dict_key}"
    xvector_dir=${xvector_dir}_teacher
fi

if [[ -n ${dinossl_cfg} ]];then
    echo "Dinossl configuration file will be read to extract xvectors"
    xvec_args+=" --dinossl_cfg ${dinossl_cfg}"
    dinossl_xvec_loc=`grep "^dinossl_xvec_loc" ${dinossl_cfg} | awk '{print $2}'`
    chkpnt_idx=${chkpnt_idx}_xloc${dinossl_xvec_loc}
fi

if [ ! -s ${nnet} ]; then
    echo "${nnet} does not exist or is wrong." && exit 1
fi


if [ $stage -le 1 ]; then
    # Extract xvectors for training LDA/PLDA
    for name in voxceleb2_train
    do
	if [ $plda_num_augs -eq 0 ]; then
    	    steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd" --nj 100 ${xvec_args} \
		--random-utt-length true --min-utt-length 400 --max-utt-length 14000 \
		--feat-config $feat_config \
    		$nnet data/${name} \
    		$xvector_dir/${name}/${chkpnt_idx}
	else
	    steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd" --nj 300 ${xvec_args} \
		--random-utt-length true --min-utt-length 400 --max-utt-length 14000 \
		--feat-config $feat_config --aug-config $plda_aug_config --num-augs $plda_num_augs \
    		$nnet data/${name} \
    		$xvector_dir/${name}_augx${plda_num_augs}/${chkpnt_idx} \
		data/${name}_augx${plda_num_augs}/${chkpnt_idx}
	fi
    done
fi


if [ $stage -le 2 ]; then
    # Extracts x-vectors for evaluation
    for name in voxceleb1_test 
    do
	num_spk=$(wc -l data/$name/spk2utt | awk '{ print $1}')
	nj=$(($num_spk < 100 ? $num_spk:100))
	steps_xvec/extract_xvectors_from_wav.sh --cmd "$xvec_cmd --mem 6G" --nj $nj ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/$name/${chkpnt_idx}
    done
fi

exit

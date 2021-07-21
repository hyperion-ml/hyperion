#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=0
config_file=default_config.sh
state_dict_key=model_teacher_state_dict # which state dict was used between student (model_state_dict) and teacher (model_teacher_state_dict for dinossl) models
dinossl_xvec_loc="f" # position we extract xvectors: choices=["f", "dinohead_mlp","dinohead_l2norm","dinohead_linear"]

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data} # plda_data == "voxceleb1_test_train"

xvector_dir=exp/xvectors/$nnet_name
chkpnt_idx=`basename ${nnet%.pth}`
if [[ ${dinossl_xvec_loc} != "f" ]]; then
    chkpnt_idx=${chkpnt_idx}_xloc${dinossl_xvec_loc}
fi

if [ "${state_dict_key}" != "model_state_dict" ];then
    echo "Scoring with x-vectors extracted from a teacher model"
    xvector_dir=${xvector_dir}_teacher
    nnet_name=${nnet_name}_teacher
fi

whole_xvec=${xvector_dir}/voxceleb1_test/${chkpnt_idx}/xvector.scp
train_xvec_dir=$xvector_dir/${plda_data}/${chkpnt_idx}
eval_xvec_dir=$xvector_dir/voxceleb1_test_eval/${chkpnt_idx}

be_dir=exp/be/$nnet_name/$be_name/${chkpnt_idx}
score_dir=exp/scores/$nnet_name/${be_name}/${chkpnt_idx}
score_plda_dir=$score_dir/plda

if [ $stage -le 0 ]; then
    if [ ! -s ${train_xvec_dir}/xvector.scp ]; then
        echo "Divide ${whole_xvec} into train: ${train_xvec_dir}/xvector.scp and eval: ${eval_xvec_dir}/xvector.scp"
        mkdir -p ${train_xvec_dir} ${eval_xvec_dir}
        if [ ! -s data/voxceleb1_test/utt2spk_train ]; then
            local/separate_vox1_trainNeval_utt2spk.py ${voxceleb1_root}/vox1_meta.csv data/voxceleb1_test/ # this generates data/voxceleb1_test/utt2spk_{train,eval}
        fi
        if [ ! -d data/voxceleb1_test_train ]; then
            utils/subset_data_dir.sh --utt-list <(awk '{print $1}' data/voxceleb1_test/utt2spk_train) data/voxceleb1_test data/voxceleb1_test_train
        fi
        if [ ! -d data/voxceleb1_test_eval ]; then
            utils/subset_data_dir.sh --utt-list <(awk '{print $1}' data/voxceleb1_test/utt2spk_eval) data/voxceleb1_test data/voxceleb1_test_eval
        fi
        if [ ! -s data/voxceleb1_test_eval/utt2model ]; then
            utils/filter_scp.pl -f 1 data/voxceleb1_test/utt2spk_eval data/voxceleb1_test/utt2model > data/voxceleb1_test_eval/utt2model
        fi
        if [ ! -s data/voxceleb1_test_eval/trials_o ]; then
            cp data/voxceleb1_test/trials_o data/voxceleb1_test_eval/trials_o
        fi
        utils/filter_scp.pl -f 1 data/voxceleb1_test/utt2spk_train ${whole_xvec} > ${train_xvec_dir}/xvector.scp
        utils/filter_scp.pl -f 1 data/voxceleb1_test/utt2spk_eval ${whole_xvec} > ${eval_xvec_dir}/xvector.scp
    fi
fi

if [ $stage -le 1 ]; then

    echo "Train PLDA on Voxceleb1 train"
    steps_be/train_be_v1.sh --cmd "$train_cmd" \
				--lda_dim $lda_dim \
				--plda_type $plda_type \
				--y_dim $plda_y_dim --z_dim $plda_z_dim \
				${train_xvec_dir}/xvector.scp \
				data/$plda_data \
				$be_dir &


    wait

fi


if [ $stage -le 2 ];then

    echo "Eval Voxceleb 1 trials_o with LDA+CentWhiten+LNorm+PLDA"
    steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
    	data/voxceleb1_test_eval/trials_o \
    	data/voxceleb1_test_eval/utt2model \
    	${eval_xvec_dir}/xvector.scp \
    	$be_dir/lda_lnorm.h5 \
    	$be_dir/plda.h5 \
    	$score_plda_dir/voxceleb1_trials_o_scores

    $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
    	local/score_voxceleb1_trials_o.sh data/voxceleb1_test_eval $score_plda_dir

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi



score_cosine_dir=exp/scores/$nnet_name/cosine/${chkpnt_idx} # NO train data
score_plda_dir=$score_cosine_dir

if [ $stage -le 3 ];then

    echo "Eval Voxceleb 1 trials_o with Cosine scoring"
    steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    	data/voxceleb1_test_eval/trials_o \
    	data/voxceleb1_test_eval/utt2model \
    	${eval_xvec_dir}/xvector.scp \
    	$score_plda_dir/voxceleb1_trials_o_scores

    $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	    local/score_voxceleb1_trials_o.sh data/voxceleb1_test_eval $score_plda_dir

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi



be_name=cw_cosine_${plda_data}
be_dir=exp/be/$nnet_name/${be_name}/${chkpnt_idx}
score_dir=exp/scores/$nnet_name/${be_name}/${chkpnt_idx}
score_plda_dir=$score_dir/cw_cosine

if [ $stage -le 4 ]; then
    echo "Train centering+whitening on Voxceleb1 train"
    steps_be/train_be_v2.sh --cmd "$train_cmd" \
	${train_xvec_dir}/xvector.scp \
	data/$plda_data \
	$be_dir
fi


if [ $stage -le 5 ];then

    echo "Eval Voxceleb 1 trials_o with CentWhiten + Cosine scoring"
    steps_be/eval_be_v2.sh --cmd "$train_cmd" \
    	data/voxceleb1_test_eval/trials_o \
    	data/voxceleb1_test_eval/utt2model \
    	${eval_xvec_dir}/xvector.scp \
    	$be_dir/cw.h5 \
    	$score_plda_dir/voxceleb1_trials_o_scores

    $train_cmd --mem 10G $score_plda_dir/log/score_voxceleb1.log \
	local/score_voxceleb1_trials_o.sh data/voxceleb1_test_eval $score_plda_dir 

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi

exit


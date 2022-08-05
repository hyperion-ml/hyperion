#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
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

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
score_cosine_dir=exp/scores/$nnet_name/cosine

score_plda_dir=$score_cosine_dir

if [ $stage -le 3 ];then

    echo "Eval Voxceleb 1 with Cosine scoring"
    steps_be/eval_be_cos.sh --cmd "$train_cmd" \
    	data/voxceleb1_test/trials \
    	data/voxceleb1_test/utt2model \
    	$xvector_dir/voxceleb1_test/xvector.scp \
    	$score_plda_dir/voxceleb1_scores

    $train_cmd --mem 20G --num-threads 6 $score_plda_dir/log/score_voxceleb1.log \
	local/score_voxceleb1.sh data/voxceleb1_test $score_plda_dir 

    for f in $(ls $score_plda_dir/*_results);
    do
	echo $f
	cat $f
	echo ""
    done

fi


exit


#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e


stage=1
nnet_stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $nnet_stage -eq 1 ]; then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ]; then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
fi


plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}
xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
score_cosine_dir=exp/scores/$nnet_name/cosine



# if [ "$do_plda" == "true" ];then
#   if [ $stage -le 1 ]; then
#     echo "Train PLDA on Voxceleb2"
#     steps_be/train_be_v1.sh \
#       --cmd "$train_cmd" \
#       --lda_dim $lda_dim \
#       --plda_type $plda_type \
#       --y_dim $plda_y_dim --z_dim $plda_z_dim \
#       $xvector_dir/$plda_data/xvector.scp \
#       data/$plda_data \
#       $be_dir
    
# fi
# fi
echo '9ala3'

if [ "$do_pca" != "true" ];then
  exit 0
fi

pca_var_r=0.99
be_name=pca_r${pca_var_r}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name

score_dir=exp/scores/$nnet_name/${be_name}
score_cosine_dir=exp/scores/$nnet_name/$be_name/cosine
score_cosine_snorm_dir=exp/scores/$nnet_name/$be_name/cosine_snorm
score_cosine_qmf_dir=exp/scores/$nnet_name/$be_name/cosine_qmf

be_dir=exp/be/$nnet_name/
score_be_dir=$score_dir/pca_r${pca_var_r}

# if [ $stage -le 10 ]; then
#   echo "Train projection on adi17"
#   $train_cmd $be_dir/log/train_be.log \
# 	     hyp_utils/conda_env.sh \
# 	     steps_be/train_be_v1.py \
# 	     --v-file scp:$xvector_dir/adi17/train_noaug/xvector.scp \
# 	     --train-list data/adi17/train/utt2lang \
# 		 --pca.pca-var-r $pca_var_r \
# 	     --output-dir $be_dir \
	   

# fi


if [ $stage -le 11 ];then
	for name in dev test 
	do
  	$train_cmd \
			${score_dir}_p12/train_${name}.log \
			hyp_utils/conda_env.sh \
			steps_be/eval_be_v2.py \
			--v-file scp:$xvector_dir/adi17/${name}/xvector.scp \
			--trial-list data/adi17/${name}_proc_audio_no_sil/utt2lang \
			--has-labels \
			--model-dir $be_dir \
			--score-file ${score_dir}adi17__${name}_cores.tsv
	done
fi
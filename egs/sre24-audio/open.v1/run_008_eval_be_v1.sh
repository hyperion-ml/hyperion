#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
nnet_stage=""
config_file=default_config.sh

# ncoh=5000
# pca_var_r=0.95
# r_mu=100
# r_s=300
# plda_type=splda
# plda_y_dim=90
# w_mu=0.75
# w_B=0.0
# w_W=0.25

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ -z "$nnet_stage" ];then
  nnet_stage=$max_nnet_stage
fi

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name

fi

#pca_label=pca${pca_var_r}_rmu${r_mu}_rs${r_s}
#plda_label=${plda_type}y${plda_y_dim}_adapt_wmu${w_mu}_wb${w_B}_ww${w_W}
#be_name=${pca_label}_${plda_label}_v2
#be_sre24_name=$be_name


xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name
be_sre21_dir=$be_dir/$be_sre21_name
be_sre24_dir=$be_dir/$be_sre24_name
score_dir=exp/scores/$nnet_name
score_plda_dir=$score_dir/${be_sre21_name}/plda
score_cosine_dir=$score_dir/cosine
score_cosine_snorm_dir=$score_dir/cosine_snorm
score_cosine_qmf_dir=$score_dir/cosine_qmf

# delete this files shouldn't be there
#rm -f data/sre21_audio-visual_{dev,eval}_test/trials_source_type_CTS_CTS.tsv

if [ $stage -le 1 ];then

  for data in sre21_audio_dev sre21_audio-visual_dev sre24_audio_dev sre24_audio-visual_dev sre24_audio_eval sre24_audio-visual_eval # sre21_audio_eval sre21_audio-visual_eval
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    
    echo "Eval $data with cosine scoring"
    (
      $train_cmd $score_cosine_dir/log/$data.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-cosine-scoring-backend \
		 --enroll-feats-file csv:$xvector_dir/$data_enroll/xvector.csv \
		 --feats-file csv:$xvector_dir/$data_test/xvector.csv \
		 --ndx-file data/$data_test/trials.tsv \
		 --enroll-map-file data/$data_enroll/enrollment.csv  \
		 --score-file $score_cosine_dir/${data}_scores.tsv 
      
      $train_cmd --mem 12G --num-threads 6 $score_cosine_dir/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/metrics_${data}.yaml \
		 --score-files $score_cosine_dir/${data}_scores.tsv \
		 --score-names $data \
		 --output-file $score_cosine_dir/${data}_results.tsv

      echo "Results $data:"
      grep -e eer -e equalized  $score_cosine_dir/${data}_results.tsv
    ) &
  done
  wait
  local/score_sre24_official.sh audio dev $score_cosine_dir &
  local/score_sre24_official.sh audio eval $score_cosine_dir
  
fi

if [ $stage -le 2 ];then
  echo "Train PLDA Adapted to SRE21 in $be_sre21_dir"
  $train_cmd $be_sre21_dir/train_plda.log \
	     python local/train_plda_source_lang_adapted.py \
	     --cfg $be_sre21_cfg \
	     --ood-segments-files data/{sre_cts_superset,sre16_eval_train,voxceleb2cat_train,sre18_cmn2_dev_enroll,sre18_cmn2_dev_test,sre18_cmn2_eval_enroll,sre18_cmn2_eval_test,sre19_cts_enroll,sre19_cts_test}/segments.csv \
	     --ood-feats-files csv:$xvector_dir/{sre_cts_superset,sre16_eval_train,voxceleb2cat_train,sre18_cmn2_dev_enroll,sre18_cmn2_dev_test,sre18_cmn2_eval_enroll,sre18_cmn2_eval_test,sre19_cts_enroll,sre19_cts_test}/xvector.csv \
	     --id-segments-files data/sre21_audio{_eval_enroll,_eval_test,-visual_eval_test}/segments.csv \
	     --id-feats-files csv:$xvector_dir/sre21_audio{_eval_enroll,_eval_test,-visual_eval_test}/xvector.csv \
	     --preproc-file $be_sre21_dir/preproc.pkl \
	     --preproc-adapt-file $be_sre21_dir/preproc_adapt.pkl \
	     --plda-file $be_sre21_dir/plda.h5 \
	     --plda-adapt-file $be_sre21_dir/plda_adapt.h5 \
	     # --pca.whiten --pca.pca-var-r $pca_var_r --pca.pca-min-dim 25 \
	     # --pca_adapt.r-mu $r_mu --pca_adapt.r-s $r_s \
	     # --plda.plda-type $plda_type --plda.y-dim $plda_y_dim \
	     # --plda_adapt.w-mu $w_mu --plda_adapt.w-B $w_B --plda_adapt.w-W $w_W \
	     # --source-types cts afv \
	     # --target-langs ENG CMN YUE \
	     # --ood-speaker-langs CMN YUE


fi


if [ $stage -le 3 ];then

  for data in sre21_audio_dev sre21_audio-visual_dev sre21_audio_eval sre21_audio-visual_eval
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    
    (echo "Eval $data with adapted PLDA in $score_plda_dir"
     $train_cmd $score_plda_dir/log/$data.log \
		hyp_utils/conda_env.sh \
		python local/eval_plda_source_lang_adapted_backend.py \
		--enroll-segments-file data/$data_enroll/segments.csv \
		--test-segments-file data/$data_test/segments.csv \
		--enroll-feats-file csv:$xvector_dir/$data_enroll/xvector.csv \
		--feats-file csv:$xvector_dir/$data_test/xvector.csv \
		--ndx-file data/$data_test/trials.tsv \
		--enroll-map-file data/$data_enroll/enrollment.csv  \
		--preproc-file $be_sre21_dir/preproc_adapt.pkl \
		--plda-file $be_sre21_dir/plda_adapt.h5 \
		--score-file $score_plda_dir/${data}_scores.tsv \
		--source-types cts afv \
		--langs ENG CMN YUE

    $train_cmd --mem 12G --num-threads 6 $score_plda_dir/log/score_${data}.log \
	       hyperion-eval-verification-metrics \
	       --cfg conf/metrics_${data}.yaml \
	       --score-files $score_plda_dir/${data}_scores.tsv \
	       --score-names $data \
	       --output-file $score_plda_dir/${data}_results.tsv

    echo "Results $data:"
    grep -e eer -e equalized $score_plda_dir/${data}_results.tsv
    ) &
  done
  wait
fi

if [ $stage -le 4 ];then
  echo "Train calibration V1 for SRE21"
  $train_cmd ${score_plda_dir}_cal_v1/log/train_calibration_v1.log \
	     hyp_utils/conda_env.sh \
	     hyperion-train-verification-calibration \
	     --score-files $score_plda_dir/sre21_{audio,audio-visual}_eval_scores.tsv \
	     --key-files data/sre21_{audio,audio-visual}_eval_test/trials.tsv \
	     --model-file ${score_plda_dir}_cal_v1/calibration.h5 \
	     --lambda-reg 1e-5
  
  for data in sre21_audio_dev sre21_audio-visual_dev sre21_audio_eval sre21_audio-visual_eval
  do
    data_test=${data}_test
    (
      echo "Eval calibration V1 for $data in ${score_plda_dir}_cal_v1"
      $train_cmd ${score_plda_dir}/log/$data.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-verification-calibration \
		 --ndx-file data/$data_test/trials.tsv \
		 --in-score-file $score_plda_dir/${data}_scores.tsv \
		 --out-score-file ${score_plda_dir}_cal_v1/${data}_scores.tsv \
		 --model-file ${score_plda_dir}_cal_v1/calibration.h5
		 
      $train_cmd --mem 12G --num-threads 3 ${score_plda_dir}_cal_v1/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/metrics_${data}.yaml \
		 --score-files ${score_plda_dir}_cal_v1/${data}_scores.tsv \
		 --score-names $data \
		 --output-file ${score_plda_dir}_cal_v1/${data}_results.tsv

      echo "Results $data:"
      grep -e eer -e equalized ${score_plda_dir}_cal_v1/${data}_results.tsv
    ) &
  done
  wait
fi

if [ $stage -le 5 ];then
  echo "Train calibration V2 for SRE21"
  $train_cmd ${score_plda_dir}_cal_v2/log/train_calibration_v2.log \
	     hyp_utils/conda_env.sh \
	     local/train_verification_calibration_v2.py \
	     --score-files $score_plda_dir/sre21_{audio,audio-visual}_eval_scores.tsv \
	     --key-files data/sre21_{audio,audio-visual}_eval_test/trials.tsv \
	     --model-file ${score_plda_dir}_cal_v2/calibration.h5 \
	     --lambda-reg 1e-5

  for data in sre21_audio_dev sre21_audio-visual_dev # sre21_audio_eval sre21_audio-visual_eval
  do
    data_test=${data}_test
    (
      echo "Eval calibration V2 for $data in ${score_plda_dir}_cal_v2"
      $train_cmd ${score_plda_dir}/log/$data.log \
		 hyp_utils/conda_env.sh \
		 local/eval_verification_calibration_v2.py \
		 --ndx-file data/$data_test/trials.tsv \
		 --in-score-file $score_plda_dir/${data}_scores.tsv \
		 --out-score-file ${score_plda_dir}_cal_v2/${data}_scores.tsv \
		 --model-file ${score_plda_dir}_cal_v2/calibration.h5
		 
      $train_cmd --mem 12G --num-threads 3 ${score_plda_dir}_cal_v2/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/metrics_${data}.yaml \
		 --score-files ${score_plda_dir}_cal_v2/${data}_scores.tsv \
		 --score-names $data \
		 --output-file ${score_plda_dir}_cal_v2/${data}_results.tsv

      echo "Results $data:"
      grep -e eer -e equalized ${score_plda_dir}_cal_v2/${data}_results.tsv
    ) &
  done
  wait
fi

score_plda_sre21_dir=$score_plda_dir
score_plda_dir=$score_dir/${be_sre24_name}/plda

if [ $stage -le 6 ];then
  echo "Train/Eval PLDA Adapted to SRE24 dev by folds"
  for fold in 0 1
  do
    
    (
      fold_be_sre24_dir=$be_sre24_dir/folds/$fold
      fold_score_plda_dir=$score_plda_dir/folds/$fold
      
      echo "Train PLDA Adapted to SRE24 fold $fold in $fold_be_sre24_dir"
      $train_cmd $fold_be_sre24_dir/train_plda.log \
		 python local/train_plda_source_lang_adapted.py \
		 --cfg $be_sre24_cfg \
		 --ood-segments-files data/{sre_cts_superset,sre16_eval_train}/segments.csv \
		 --ood-feats-files csv:$xvector_dir/{sre_cts_superset,sre16_eval_train}/xvector.csv \
		 --id-segments-files data/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/folds/$fold/train/segments.csv \
		 data/sre18_cmn2_{dev,eval}_{enroll,test}/segments.csv \
		 data/sre19_cts_{enroll,test}/segments.csv \
		 --id-feats-files csv:$xvector_dir/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/xvector.csv \
		 csv:$xvector_dir/sre18_cmn2_{dev,eval}_{enroll,test}/xvector.csv \
		 csv:$xvector_dir/sre19_cts_{enroll,test}/xvector.csv \
		 --preproc-file $fold_be_sre24_dir/preproc.pkl \
		 --preproc-adapt-file $fold_be_sre24_dir/preproc_adapt.pkl \
		 --plda-file $fold_be_sre24_dir/plda.h5 \
		 --plda-adapt-file $fold_be_sre24_dir/plda_adapt.h5 \
		 # --pca.whiten --pca.pca-var-r $pca_var_r --pca.pca-min-dim 25 \
		 # --pca_adapt.r-mu $r_mu --pca_adapt.r-s $r_s \
		 # --plda.plda-type $plda_type --plda.y-dim $plda_y_dim \
		 # --plda_adapt.w-mu $w_mu --plda_adapt.w-B $w_B --plda_adapt.w-W $w_W \
		 # --source-types cts afv \
		 # --target-langs ENG ARA FRA \
		 # --ood-speaker-langs ARA FRA

		   
      for data in sre24_audio_dev sre24_audio-visual_dev
      do
	data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
	data_test=${data}_test
	fold_data_enroll=$data_enroll/folds/$fold/test
	fold_data_test=$data_test/folds/$fold/test
	
	echo "Eval $data with adapted PLDA in $fold_score_plda_dir"
	(
	  $train_cmd $fold_score_plda_dir/log/$data.log \
		     hyp_utils/conda_env.sh \
		     python local/eval_plda_source_lang_adapted_backend.py \
		     --enroll-segments-file data/$fold_data_enroll/segments.csv \
		     --test-segments-file data/$fold_data_test/segments.csv \
		     --enroll-feats-file csv:$xvector_dir/$data_enroll/xvector.csv \
		     --feats-file csv:$xvector_dir/$data_test/xvector.csv \
		     --ndx-file data/$fold_data_test/trials.csv \
		     --enroll-map-file data/$fold_data_enroll/enrollment.csv  \
		     --preproc-file $fold_be_sre24_dir/preproc_adapt.pkl \
		     --plda-file $fold_be_sre24_dir/plda_adapt.h5 \
		     --score-file $fold_score_plda_dir/${data}_scores.tsv \
		     --source-types cts afv \
		     --langs ENG ARA FRA

	  # $train_cmd --mem 12G --num-threads 6 $fold_score_plda_dir/log/score_${data}.log \
	  # 	     hyperion-eval-verification-metrics \
	  # 	     --cfg conf/folds/$fold/metrics_${data}.yaml \
	  # 	     --score-files $fold_score_plda_dir/${data}_scores.tsv \
	  # 	     --score-names $data \
	  # 	     --output-file $fold_score_plda_dir/${data}_results.tsv

	  #echo "Results $data:"
	  #cat $fold_score_plda_dir/${data}_results.tsv

	) &
	
      done
      wait
    ) &
  done
  wait
  for data in sre24_audio_dev sre24_audio-visual_dev
  do
    (
      # hyperion-tables average_results \
      # 		      --input-files $score_plda_dir/folds/{0,1}/${data}_results.tsv \
      # 		      --output-file $score_plda_dir/folds/avg/${data}_results.tsv
      # echo "Results $data avg folds:"
      # cat $score_plda_dir/folds/avg/${data}_results.tsv
      hyperion-merge-scores --input-files $score_plda_dir/folds/{0,1}/${data}_scores.tsv \
			    --output-file $score_plda_dir/folds/0+1/${data}_scores.tsv
      $train_cmd --mem 12G --num-threads 6 $score_plda_dir/folds/0+1/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/folds/0+1/metrics_${data}.yaml \
		 --score-files $score_plda_dir/folds/0+1/${data}_scores.tsv \
		 --score-names $data \
		 --output-file $score_plda_dir/folds/0+1/${data}_results.tsv
      echo "Results $data merged folds:"
      grep -e eer -e equalized $score_plda_dir/folds/0+1/${data}_results.tsv
    ) &
  done
  wait

fi

if [ $stage -le 7 ];then
  echo "Train PLDA Adapted to SRE24 in $be_sre24_dir"
  $train_cmd $be_sre24_dir/train_plda.log \
	     python local/train_plda_source_lang_adapted.py \
	     --cfg $be_sre24_cfg \
	     --ood-segments-files data/{sre_cts_superset,sre16_eval_train}/segments.csv \
	     --ood-feats-files csv:$xvector_dir/{sre_cts_superset,sre16_eval_train}/xvector.csv \
	     --id-segments-files data/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/segments.csv \
	     data/sre18_cmn2_{dev,eval}_{enroll,test}/segments.csv \
	     data/sre19_cts_{enroll,test}/segments.csv \
	     --id-feats-files csv:$xvector_dir/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/xvector.csv \
	     csv:$xvector_dir/sre18_cmn2_{dev,eval}_{enroll,test}/xvector.csv \
	     csv:$xvector_dir/sre19_cts_{enroll,test}/xvector.csv \
	     --preproc-file $be_sre24_dir/preproc.pkl \
	     --preproc-adapt-file $be_sre24_dir/preproc_adapt.pkl \
	     --plda-file $be_sre24_dir/plda.h5 \
	     --plda-adapt-file $be_sre24_dir/plda_adapt.h5 \
	     # --pca.whiten --pca.pca-var-r $pca_var_r --pca.pca-min-dim 25 \
	     # --pca_adapt.r-mu $r_mu --pca_adapt.r-s $r_s \
	     # --plda.plda-type $plda_type --plda.y-dim $plda_y_dim \
	     # --plda_adapt.w-mu $w_mu --plda_adapt.w-B $w_B --plda_adapt.w-W $w_W \
	     # --source-types cts afv \
	     # --target-langs ENG ARA FRA \
	     # --ood-speaker-langs ARA FRA
fi


if [ $stage -le 8 ];then

  for data in sre24_audio_dev sre24_audio-visual_dev sre24_audio_eval sre24_audio-visual_eval
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    
    echo "Eval $data with adapted PLDA in $score_plda_dir"
    $train_cmd $score_plda_dir/log/$data.log \
	       hyp_utils/conda_env.sh \
	       python local/eval_plda_source_lang_adapted_backend.py \
	       --enroll-segments-file data/$data_enroll/segments.csv \
	       --test-segments-file data/$data_test/segments.csv \
	       --enroll-feats-file csv:$xvector_dir/$data_enroll/xvector.csv \
	       --feats-file csv:$xvector_dir/$data_test/xvector.csv \
	       --ndx-file data/$data_test/trials.tsv \
	       --enroll-map-file data/$data_enroll/enrollment.csv  \
	       --preproc-file $be_sre24_dir/preproc_adapt.pkl \
	       --plda-file $be_sre24_dir/plda_adapt.h5 \
	       --score-file $score_plda_dir/${data}_scores.tsv \
	       --source-types cts afv \
	       --langs ENG ARA FRA

    $train_cmd --mem 12G --num-threads 6 $score_plda_dir/log/score_${data}.log \
	       hyperion-eval-verification-metrics \
	       --cfg conf/metrics_${data}.yaml \
	       --score-files $score_plda_dir/${data}_scores.tsv \
	       --score-names $data \
	       --output-file $score_plda_dir/${data}_results.tsv

    echo "Results $data:"
    grep -e eer -e equalized  $score_plda_dir/${data}_results.tsv
  done
fi


if [ $stage -le 9 ];then
  echo "Train calibration V1 for SRE24 on folds scores"
  score_plda_cal_dir=${score_plda_dir}_cal_v1_folds
  $train_cmd ${score_plda_cal_dir}/log/train_calibration_sre24_v1.log \
	     hyp_utils/conda_env.sh \
	     hyperion-train-verification-calibration \
	     --score-files $score_plda_dir/folds/0+1/sre24_{audio,audio-visual}_dev_scores.tsv \
	     --key-files data/sre24_{audio,audio-visual}_dev_test/folds/0+1/test/trials.csv \
	     --model-file ${score_plda_cal_dir}/calibration.h5 \
	     --lambda-reg 1e-5 --prior 0.01 --num-augs 10 10 --aug-std 5
      
  for data in sre24_audio_dev sre24_audio-visual_dev sre24_audio_eval sre24_audio-visual_eval
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    (
      echo "Eval calibration V1 folds for $data in ${score_plda_dir}_cal_v1"
      $train_cmd ${score_plda_cal_dir}/log/$data.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-verification-calibration \
		 --ndx-file data/$data_test/trials.csv \
		 --in-score-file $score_plda_dir/${data}_scores.tsv \
		 --out-score-file ${score_plda_cal_dir}/${data}_scores.tsv \
		 --model-file ${score_plda_cal_dir}/calibration.h5
	  
      $train_cmd --mem 12G --num-threads 3 $score_plda_cal_dir/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/metrics_${data}.yaml \
		 --score-files $score_plda_cal_dir/${data}_scores.tsv \
		 --score-names $data \
		 --output-file $score_plda_cal_dir/${data}_results.tsv

      echo "Results $data: $score_plda_cal_dir/${data}_results.tsv"
      grep -e eer -e equalized $score_plda_cal_dir/${data}_results.tsv
    ) &
  done
  wait
  fold_score_plda_dir=$score_plda_dir/folds/0+1
  fold_score_plda_cal_dir=$score_plda_cal_dir/folds/0+1
  for data in sre24_audio_dev sre24_audio-visual_dev
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    fold_data_enroll=$data_enroll/folds/0+1/test
    fold_data_test=$data_test/folds/0+1/test
    (
      echo "Eval calibration V1 for $data in ${fold_score_plda_cal_dir}"
      $train_cmd ${fold_score_plda_cal_dir}/log/$data.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-verification-calibration \
		 --ndx-file data/$fold_data_test/trials.tsv \
		 --in-score-file $fold_score_plda_dir/${data}_scores.tsv \
		 --out-score-file ${fold_score_plda_cal_dir}/${data}_scores.tsv \
		 --model-file ${score_plda_cal_dir}/calibration.h5

      $train_cmd --mem 12G --num-threads 6 $score_plda_cal_dir/folds/0+1/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/folds/0+1/metrics_${data}.yaml \
		 --score-files $fold_score_plda_cal_dir/${data}_scores.tsv \
		 --score-names $data \
		 --output-file $fold_score_plda_cal_dir/${data}_results.tsv
      echo "Results $data merged $fold_score_plda_cal_dir/${data}_results.tsv"
      grep -e eer -e equalized $fold_score_plda_cal_dir/${data}_results.tsv 
    ) &
  done
  wait
  

fi

if [ $stage -le 10 ];then
  echo "Train calibration V2 for SRE24 on folds scores"
  score_plda_cal_dir=${score_plda_dir}_cal_v2_folds
  $train_cmd ${score_plda_cal_dir}/log/train_calibration_sre24_v2.log \
	     hyp_utils/conda_env.sh \
	     local/train_verification_calibration_v2.py \
	     --score-files $score_plda_dir/folds/0+1/sre24_{audio,audio-visual}_dev_scores.tsv \
	     --key-files data/sre24_{audio,audio-visual}_dev_test/folds/0+1/test/trials.tsv \
	     --model-file ${score_plda_cal_dir}/calibration.h5 \
	     --lambda-reg 1e-5 --prior 0.01 --num-augs 10 10 --aug-std 0.5
      
  for data in sre24_audio_dev sre24_audio-visual_dev sre24_audio_eval sre24_audio-visual_eval
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    (
      echo "Eval calibration V2 folds for $data in ${score_plda_cal_dir}"
      $train_cmd ${score_plda_cal_dir}/log/$data.log \
		     hyp_utils/conda_env.sh \
		     local/eval_verification_calibration_v2.py \
		     --ndx-file data/$data_test/trials.tsv \
		     --in-score-file $score_plda_dir/${data}_scores.tsv \
		     --out-score-file ${score_plda_cal_dir}/${data}_scores.tsv \
		     --model-file ${score_plda_cal_dir}/calibration.h5
	  
      $train_cmd --mem 12G --num-threads 3 $score_plda_cal_dir/log/score_${data}.log \
		     hyperion-eval-verification-metrics \
		     --cfg conf/metrics_${data}.yaml \
		     --score-files $score_plda_cal_dir/${data}_scores.tsv \
		     --score-names $data \
		     --output-file $score_plda_cal_dir/${data}_results.tsv

      echo "Results $data: $score_plda_cal_dir/${data}_results.tsv"
      grep -e eer -e equalized $score_plda_cal_dir/${data}_results.tsv
    ) &
  done
  wait
  fold_score_plda_dir=$score_plda_dir/folds/0+1
  fold_score_plda_cal_dir=$score_plda_cal_dir/folds/0+1
  for data in sre24_audio_dev sre24_audio-visual_dev
  do
    data_enroll=$(echo ${data}_enroll | sed 's@audio-visual@audio@')
    data_test=${data}_test
    fold_data_enroll=$data_enroll/folds/0+1/test
    fold_data_test=$data_test/folds/0+1/test
    (
      echo "Eval calibration V2 for $data in ${fold_score_plda_cal_dir}"
      $train_cmd ${fold_score_plda_cal_dir}/log/$data.log \
		 hyp_utils/conda_env.sh \
		 local/eval_verification_calibration_v2.py \
		 --ndx-file data/$fold_data_test/trials.tsv \
		 --in-score-file $fold_score_plda_dir/${data}_scores.tsv \
		 --out-score-file ${fold_score_plda_cal_dir}/${data}_scores.tsv \
		 --model-file ${score_plda_cal_dir}/calibration.h5

      $train_cmd --mem 12G --num-threads 6 $score_plda_cal_dir/folds/0+1/log/score_${data}.log \
		 hyperion-eval-verification-metrics \
		 --cfg conf/folds/0+1/metrics_${data}.yaml \
		 --score-files $fold_score_plda_cal_dir/${data}_scores.tsv \
		 --score-names $data \
		 --output-file $fold_score_plda_cal_dir/${data}_results.tsv
      echo "Results $data merged $fold_score_plda_cal_dir/${data}_results.tsv"
      grep -e eer -e equalized $fold_score_plda_cal_dir/${data}_results.tsv 
    ) &
  done
  wait
fi


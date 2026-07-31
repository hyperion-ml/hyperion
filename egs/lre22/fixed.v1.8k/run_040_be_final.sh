#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=2
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file

if [ $nnet_stages -lt $nnet_stage ];then
    nnet_stage=$nnet_stages
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
elif [ $nnet_stage -eq 4 ];then
  nnet=$nnet_s4
  nnet_name=$nnet_s4_name
elif [ $nnet_stage -eq 5 ];then
  nnet=$nnet_s5
  nnet_name=$nnet_s5_name
fi

xvector_dir=exp/xvectors/$nnet_name
be_base_dir=exp/be/$nnet_name
score_base_dir=exp/scores/$nnet_name

if [ $stage -le 1 ];then
  for r in 1 #0.9999 0.999 #0.99 0.975 0.95
  do
    be_name=pca${r}_cw_lnorm_lgbe_lre22_aug
    be_dir=$be_base_dir/$be_name
    score_dir=$score_base_dir/$be_name

    (
      for p_trn in p1 p2
      do

	if [ "$p_trn" == "p1" ];then
	  p_test="p2"
	else
	  p_test="p1"
	fi
	be_dir_p=${be_dir}_$p_trn
	(
	  $train_cmd \
	    $be_dir_p/train.log \
	    hyp_utils/conda_env.sh \
	    steps_be/train_be_v1.py \
	    --v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
	    --train-list data/lre22_dev_aug_clean_$p_trn/utt2lang \
	    --pca.pca-var-r $r \
	    --do-lnorm --whiten \
	    --output-dir $be_dir_p

	  $train_cmd \
	    ${score_dir}_p12/test_${p_test}.log \
	    hyp_utils/conda_env.sh \
	    steps_be/eval_be_v1.py \
	    --v-file scp:$xvector_dir/lre22_dev/xvector.scp \
	    --trial-list data/lre22_dev_$p_test/utt2lang \
	    --has-labels \
	    --model-dir $be_dir_p \
	    --score-file ${score_dir}_p12/nocal/lre22_dev_${p_test}_scores.tsv


	) &
	
      done

      (
	$train_cmd \
	  $be_dir/train.log \
	  hyp_utils/conda_env.sh \
	  steps_be/train_be_v1.py \
	  --v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
	  --train-list data/lre22_dev_aug_clean/utt2lang \
	  --pca.pca-var-r $r \
	  --do-lnorm --whiten \
	  --output-dir $be_dir

	$train_cmd \
	    ${score_dir}_p12/test_dev.log \
	    hyp_utils/conda_env.sh \
	    steps_be/eval_be_v1.py \
	    --v-file scp:$xvector_dir/lre22_dev/xvector.scp \
	    --trial-list data/lre22_dev/utt2lang \
	    --has-labels \
	    --model-dir $be_dir \
	    --score-file ${score_dir}/nocal/lre22_dev_scores.tsv

	$train_cmd \
	    ${score_dir}/test_eval.log \
	    hyp_utils/conda_env.sh \
	    steps_be/eval_be_v1.py \
	    --v-file scp:$xvector_dir/lre22_eval/xvector.scp \
	    --trial-list data/lre22_eval/utt2spk \
	    --model-dir $be_dir \
	    --score-file ${score_dir}/nocal/lre22_eval_scores.tsv

	) &

      wait

      hyp_utils/conda_env.sh \
	local/merge_scores.py \
	--in-score-files ${score_dir}_p12/nocal/lre22_dev_p{1,2}_scores.tsv \
	--out-score-file ${score_dir}_p12/nocal/lre22_dev_scores.tsv

      local/score_lre22.sh dev \
	${score_dir}_p12/nocal/lre22_dev_scores.tsv \
	${score_dir}_p12/nocal/lre22_dev_results

      local/train_calibration_lre22.sh ${score_dir}_p12
      local/score_lre22.sh dev \
	${score_dir}_p12/cal_v1/lre22_dev_scores.tsv \
	${score_dir}_p12/cal_v1/lre22_dev_results

      local/score_lre22.sh dev \
	${score_dir}/nocal/lre22_dev_scores.tsv \
	${score_dir}/nocal/lre22_dev_results
      local/score_lre22.sh eval \
	${score_dir}/nocal/lre22_eval_scores.tsv \
	${score_dir}/nocal/lre22_eval_results

      local/eval_calibration_lre22.sh $score_dir ${score_dir}_p12/cal_v1/cal.mat
      local/score_lre22.sh dev \
	${score_dir}/cal_v1/lre22_dev_scores.tsv \
	${score_dir}/cal_v1/lre22_dev_results
      local/score_lre22.sh eval \
	${score_dir}/cal_v1/lre22_eval_scores.tsv \
	${score_dir}/cal_v1/lre22_eval_results

      # local/validate_lre22.sh \
      # 	${score_dir}/cal_v1/lre22_eval_scores.tsv

     ) &

    
  done
  wait

fi

exit
# Back-ends below over-fitted

if [ $stage -le 2 ];then
  for r in 1 
  do
    for penalty in l2 #l1
    do
      for c in 1 #0.1 1
      do
	for ary_thr in 0.975 #0.85 0.7 #0.99 0.95 0.9 #15 ##1 5 10 20
	do
	  be_name=pca${r}_cw_lnorm_lsvm_${penalty}_c${c}_sqhinge_lre22_aug_lre17_aryt${ary_thr}
	  be_dir=$be_base_dir/$be_name
	  score_dir=$score_base_dir/$be_name
	  (
	    for p_trn in p1 p2
	    do
	      
	      if [ "$p_trn" == "p1" ];then
		p_test="p2"
	      else
		p_test="p1"
	      fi
	      
	      be_dir_p=${be_dir}_$p_trn
	      (
		$train_cmd \
		  $be_dir_p/train.log \
		  hyp_utils/conda_env.sh \
		  steps_be/train_be_v3.py \
		  --v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
		  --train-list data/lre22_dev_aug_clean_$p_trn/utt2lang \
		  --lre17-v-file scp:$xvector_dir/lre17_proc_audio_no_sil/xvector.scp \
		  --lre17-list data/lre17_proc_audio_no_sil/utt2lang \
		  --pca.pca-var-r $r \
		  --svm.penalty $penalty --svm.c $c --svm.dual false \
		  --do-lnorm --whiten --ary-thr $ary_thr \
		  --output-dir $be_dir_p
		
		$train_cmd \
		  ${score_dir}_p12/test_${p_test}.log \
		  hyp_utils/conda_env.sh \
		  steps_be/eval_be_v2.py \
		  --v-file scp:$xvector_dir/lre22_dev/xvector.scp \
		  --trial-list data/lre22_dev_$p_test/utt2lang \
		  --has-labels \
		  --model-dir $be_dir_p \
		  --score-file ${score_dir}_p12/nocal/lre22_dev_${p_test}_scores.tsv
	      ) &
	    done
	    (
	      $train_cmd \
		$be_dir/train.log \
		hyp_utils/conda_env.sh \
		steps_be/train_be_v3.py \
		--v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
		--train-list data/lre22_dev_aug_clean/utt2lang \
		--lre17-v-file scp:$xvector_dir/lre17_proc_audio_no_sil/xvector.scp \
		--lre17-list data/lre17_proc_audio_no_sil/utt2lang \
		--pca.pca-var-r $r \
		--svm.penalty $penalty --svm.c $c --svm.dual false \
		--do-lnorm --whiten --ary-thr $ary_thr \
		--output-dir $be_dir
		
	      $train_cmd \
		${score_dir}/test_dev.log \
		hyp_utils/conda_env.sh \
		steps_be/eval_be_v2.py \
		--v-file scp:$xvector_dir/lre22_dev/xvector.scp \
		--trial-list data/lre22_dev/utt2lang \
		--has-labels \
		--model-dir $be_dir \
		--score-file ${score_dir}/nocal/lre22_dev_scores.tsv

	      $train_cmd \
		${score_dir}/test_eval.log \
		hyp_utils/conda_env.sh \
		steps_be/eval_be_v2.py \
		--v-file scp:$xvector_dir/lre22_eval/xvector.scp \
		--trial-list data/lre22_eval/utt2spk \
		--model-dir $be_dir \
		--score-file ${score_dir}/nocal/lre22_eval_scores.tsv

	    ) &
	    
	    wait
	    hyp_utils/conda_env.sh \
	      local/merge_scores.py \
	      --in-score-files ${score_dir}_p12/nocal/lre22_dev_p{1,2}_scores.tsv \
	      --out-score-file ${score_dir}_p12/nocal/lre22_dev_scores.tsv
	  
	    local/score_lre22.sh \
	      dev \
	      ${score_dir}_p12/nocal/lre22_dev_scores.tsv \
	      ${score_dir}_p12/nocal/lre22_dev_results
	    
	    local/train_calibration_lre22.sh ${score_dir}_p12
	    local/score_lre22.sh \
	      dev \
	      ${score_dir}_p12/cal_v1/lre22_dev_scores.tsv \
	      ${score_dir}_p12/cal_v1/lre22_dev_results

	    local/score_lre22.sh \
	      dev \
	      ${score_dir}/nocal/lre22_dev_scores.tsv \
	      ${score_dir}/nocal/lre22_dev_results
	    local/score_lre22.sh \
	      eval \
	      ${score_dir}/nocal/lre22_eval_scores.tsv \
	      ${score_dir}/nocal/lre22_eval_results


	    local/eval_calibration_lre22.sh $score_dir ${score_dir}_p12/cal_v1/cal.mat
	    local/score_lre22.sh \
	      dev \
	      ${score_dir}/cal_v1/lre22_dev_scores.tsv \
	      ${score_dir}/cal_v1/lre22_dev_results
	    local/score_lre22.sh \
	      eval \
	      ${score_dir}/cal_v1/lre22_eval_scores.tsv \
	      ${score_dir}/cal_v1/lre22_eval_results

	    # local/validate_lre22.sh \
	    #   ${score_dir}/cal_v1/lre22_eval_scores.tsv
	    
	  ) &
	done
      done
    done
  done
  wait

fi

if [ $stage -le 3 ];then
  for r in 1 # 0.9999 0.99 0.975 0.95 0.9 0.8
  do
    for shrinking in true #false
    do
      for c in 1 10 #0.1 1 10 #0.01 0.1 1 10 # 0.0001
      do
	for vl in false #true #false
	do
	  if [ "$vl" == "true" ];then
	    do_vl="--do-vl"
	  else
	    do_vl="--no_do-vl"
	  fi
	  ary_thr=0.975
	  be_name=pca${r}_cw_lnorm_gsvm_shrinking_${shrinking}_c${c}_lre17_aryt${ary_thr}_vl${vl}_aug_clean
	  be_dir=$be_base_dir/$be_name
	  score_dir=$score_base_dir/$be_name
	  #score_dir=$score_base_dir/${be_name}_logpost
	  (
	    for p_trn in p1 p2
	    do

	      if [ "$p_trn" == "p1" ];then
		p_test="p2"
	      else
		p_test="p1"
	      fi

	      be_dir_p=${be_dir}_$p_trn
	      (
		$train_cmd $be_dir_p/train.log \
			   hyp_utils/conda_env.sh \
			   steps_be/train_be_v5.py \
			   --v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
			   --train-list data/lre22_dev_aug_clean_$p_trn/utt2lang \
			   --lre17-v-file scp:$xvector_dir/lre17_proc_audio_no_sil/xvector.scp \
			   --lre17-list data/lre17_proc_audio_no_sil/utt2lang \
			   --voxlingua-v-file scp:$xvector_dir/voxlingua107_codecs_proc_audio_no_sil/xvector.scp \
			   --voxlingua-list data/voxlingua107_codecs_proc_audio_no_sil/utt2lang \
			   --pca.pca-var-r $r \
			   --svm.shrinking $shrinking --svm.c $c --svm.break_ties false --svm.max-iter 500\
			   --do-lnorm --whiten --ary-thr $ary_thr \
			   --output-dir $be_dir_p \
			   --do-lre17 $do_vl

		$train_cmd ${score_dir}_p12/test_${p_test}.log \
			   hyp_utils/conda_env.sh \
			   steps_be/eval_be_v5.py \
			   --v-file scp:$xvector_dir/lre22_dev/xvector.scp \
			   --trial-list data/lre22_dev_$p_test/utt2lang \
			   --svm.eval-type cat-log-post \
			   --has-labels \
			   --model-dir $be_dir_p \
			   --score-file ${score_dir}_p12/nocal/lre22_dev_${p_test}_scores.tsv
	      ) &
	    done
	    (
		$train_cmd $be_dir/train.log \
			   hyp_utils/conda_env.sh \
			   steps_be/train_be_v5.py \
			   --v-file scp:$xvector_dir/lre22_dev_aug_clean/xvector.scp \
			   --train-list data/lre22_dev_aug_clean/utt2lang \
			   --lre17-v-file scp:$xvector_dir/lre17_proc_audio_no_sil/xvector.scp \
			   --lre17-list data/lre17_proc_audio_no_sil/utt2lang \
			   --voxlingua-v-file scp:$xvector_dir/voxlingua107_codecs_proc_audio_no_sil/xvector.scp \
			   --voxlingua-list data/voxlingua107_codecs_proc_audio_no_sil/utt2lang \
			   --pca.pca-var-r $r \
			   --svm.shrinking $shrinking --svm.c $c --svm.break_ties false --svm.max-iter 500 \
			   --do-lnorm --whiten --ary-thr $ary_thr \
			   --output-dir $be_dir \
			   --do-lre17 $do_vl

		$train_cmd ${score_dir}/test_dev.log \
			   hyp_utils/conda_env.sh \
			   steps_be/eval_be_v5.py \
			   --v-file scp:$xvector_dir/lre22_dev/xvector.scp \
			   --trial-list data/lre22_dev/utt2lang \
			   --svm.eval-type cat-log-post \
			   --has-labels \
			   --model-dir $be_dir \
			   --score-file ${score_dir}/nocal/lre22_dev_scores.tsv
		
		$train_cmd ${score_dir}/test_eval.log \
			   hyp_utils/conda_env.sh \
			   steps_be/eval_be_v5.py \
			   --v-file scp:$xvector_dir/lre22_eval/xvector.scp \
			   --trial-list data/lre22_eval/utt2spk \
			   --svm.eval-type cat-log-post \
			   --model-dir $be_dir \
			   --score-file ${score_dir}/nocal/lre22_eval_scores.tsv

	      ) &

	    wait
	    hyp_utils/conda_env.sh \
	      local/merge_scores.py \
	      --in-score-files ${score_dir}_p12/nocal/lre22_dev_p{1,2}_scores.tsv \
	      --out-score-file ${score_dir}_p12/nocal/lre22_dev_scores.tsv

	    local/score_lre22.sh \
	      dev \
	      ${score_dir}_p12/nocal/lre22_dev_scores.tsv \
	      ${score_dir}_p12/nocal/lre22_dev_results

	    local/train_calibration_lre22.sh ${score_dir}_p12
	    local/score_lre22.sh \
	      dev \
	      ${score_dir}_p12/cal_v1/lre22_dev_scores.tsv \
	      ${score_dir}_p12/cal_v1/lre22_dev_results

	    local/score_lre22.sh \
	      dev \
	      ${score_dir}/nocal/lre22_dev_scores.tsv \
	      ${score_dir}/nocal/lre22_dev_results
	    local/score_lre22.sh \
	      eval \
	      ${score_dir}/nocal/lre22_eval_scores.tsv \
	      ${score_dir}/nocal/lre22_eval_results

	    local/eval_calibration_lre22.sh $score_dir ${score_dir}_p12/cal_v1/cal.mat
	    local/score_lre22.sh \
	      dev \
	      ${score_dir}/cal_v1/lre22_dev_scores.tsv \
	      ${score_dir}/cal_v1/lre22_dev_results
	    local/score_lre22.sh \
	      eval \
	      ${score_dir}/cal_v1/lre22_eval_scores.tsv \
	      ${score_dir}/cal_v1/lre22_eval_results

	    # local/validate_lre22.sh \
	    #   ${score_dir}/cal_v1/lre22_eval_scores.tsv


	  ) &
	done
      done
    done
  done
  wait

fi

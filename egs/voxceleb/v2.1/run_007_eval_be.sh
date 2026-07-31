#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=3
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh 

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
elif [ $nnet_stage -eq 6 ];then
  nnet=$nnet_s6
  nnet_name=$nnet_s6_name
fi

plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}

xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name
score_plda_dir=$score_dir/${be_name}/plda
score_cosine_dir=$score_dir/cosine
score_cosine_snorm_dir=$score_dir/cosine_snorm
score_cosine_qmf_dir=$score_dir/cosine_qmf

if [ $stage -le 3 ];then

  echo "Eval Voxceleb 1 with Cosine scoring"
  num_parts=8
  for((i=1;i<=$num_parts;i++));
  do
    for((j=1;j<=$num_parts;j++));
    do
      $train_cmd $score_cosine_dir/log/voxceleb1_${i}_${j}.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-cosine-scoring-backend \
		 --feats-file csv:$xvector_dir/voxceleb1_test/xvector.csv \
		 --ndx-file data/voxceleb1_test/trials.csv \
		 --enroll-map-file data/voxceleb1_test/enrollment.csv  \
		 --score-file $score_cosine_dir/voxceleb1_scores.csv \
		 --enroll-part-idx $i --num-enroll-parts $num_parts \
		 --test-part-idx $j --num-test-parts $num_parts &
    done
  done
  wait
  hyperion-merge-scores --output-file $score_cosine_dir/voxceleb1_scores.csv \
			--num-enroll-parts $num_parts --num-test-parts $num_parts

  $train_cmd --mem 12G --num-threads 6 $score_cosine_dir/log/score_voxceleb1.log \
	     hyperion-eval-verification-metrics \
	     --score-files $score_cosine_dir/voxceleb1_scores.csv \
	     --key-files data/voxceleb1_test/trials_{o,e,h}.csv \
	     --score-names voxceleb1 \
	     --key-names O E H \
	     --sparse \
	     --output-file $score_cosine_dir/voxceleb1_results.csv

  cat $score_cosine_dir/voxceleb1_results.csv
fi

if [ $stage -le 4 ] && [ "$do_voxsrc22" == "true" ];then
  echo "Eval voxsrc2 with Cosine scoring"
  $train_cmd $score_cosine_dir/log/voxsrc22_dev.log \
	     hyp_utils/conda_env.sh \
	     hyperion-eval-cosine-scoring-backend \
	     --feats-file csv:$xvector_dir/voxsrc22_dev/xvector.csv \
	     --ndx-file data/voxsrc22_dev/trials.csv \
	     --enroll-map-file data/voxsrc22_dev/enrollment.csv  \
	     --score-file $score_cosine_dir/voxsrc22_dev_scores.csv

  # $train_cmd $score_cosine_dir/log/voxsrc22_eval.log \
    # 	     hyp_utils/conda_env.sh \
    # 	     hyperion-eval-cosine-scoring-backend \
    # 	     --feats-file csv:$xvector_dir/voxsrc22_eval/xvector.csv \
    # 	     --ndx-file data/voxsrc22_eval/trials.csv \
    # 	     --enroll-map-file data/voxsrc22_eval/enrollment.csv  \
    # 	     --score-file $score_cosine_dir/voxsrc22_eval_scores.csv
  
  $train_cmd --mem 12G --num-threads 6 $score_cosine_dir/log/score_voxsrc22_dev.log \
	     hyperion-eval-verification-metrics \
	     --score-files $score_cosine_dir/voxsrc22_dev_scores.csv \
	     --key-files data/voxsrc22_dev/trials.csv \
	     --score-names voxsrc22_dev \
	     --key-names all \
	     --output-file $score_cosine_dir/voxsrc22_dev_results.csv

  cat $score_cosine_dir/voxsrc22_dev_results.csv

fi

if [ "$do_snorm" == "true" ];then
  if [ $stage -le 5 ];then
    echo "Eval Voxceleb 1 with Cosine scoring + Adaptive SNorm"
    num_parts=16
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do
	$train_cmd --mem 22G $score_cosine_snorm_dir/log/voxceleb1_${i}_${j}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-cosine-scoring-backend \
		   --feats-file csv:$xvector_dir/voxceleb1_test/xvector.csv \
		   --ndx-file data/voxceleb1_test/trials.csv \
		   --enroll-map-file data/voxceleb1_test/enrollment.csv  \
		   --score-file $score_cosine_snorm_dir/voxceleb1_scores.csv \
		   --cohort-segments-file data/voxceleb2cat_train_cohort/segments.csv \
		   --cohort-feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --cohort-nbest 1000 --avg-cohort-by speaker \
		   --enroll-part-idx $i --num-enroll-parts $num_parts \
		   --test-part-idx $j --num-test-parts $num_parts &
      done
      sleep 5s
    done
    wait
    hyperion-merge-scores --output-file $score_cosine_snorm_dir/voxceleb1_scores.csv \
			  --num-enroll-parts $num_parts --num-test-parts $num_parts
    
    $train_cmd --mem 12G --num-threads 6 $score_cosine_snorm_dir/log/score_voxceleb1.log \
	       hyperion-eval-verification-metrics \
	       --score-files $score_cosine_snorm_dir/voxceleb1_scores.csv \
	       --key-files data/voxceleb1_test/trials_{o,e,h}.csv \
	       --score-names voxceleb1 \
	       --key-names O E H \
	       --sparse \
	       --output-file $score_cosine_snorm_dir/voxceleb1_results.csv
    
    cat $score_cosine_snorm_dir/voxceleb1_results.csv
  fi

  if [ $stage -le 6 ] && [ "$do_voxsrc22" == "true" ];then
    echo "Eval voxsrc2 with Cosine scoring + AS-Norm"
    num_parts=16
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do    
	$train_cmd $score_cosine_snorm_dir/log/voxsrc22_dev_${i}_${j}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-cosine-scoring-backend \
		   --feats-file csv:$xvector_dir/voxsrc22_dev/xvector.csv \
		   --ndx-file data/voxsrc22_dev/trials.csv \
		   --enroll-map-file data/voxsrc22_dev/enrollment.csv  \
		   --score-file $score_cosine_snorm_dir/voxsrc22_dev_scores.csv \
		   --cohort-segments-file data/voxceleb2cat_train_cohort/segments.csv \
		   --cohort-feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --cohort-nbest 1000 --avg-cohort-by speaker \
		   --enroll-part-idx $i --num-enroll-parts $num_parts \
		   --test-part-idx $j --num-test-parts $num_parts &
	sleep 5s
      done
      sleep 10s
    done
    wait
    hyperion-merge-scores --output-file $score_cosine_snorm_dir/voxsrc22_dev_scores.csv \
			  --num-enroll-parts $num_parts --num-test-parts $num_parts

    $train_cmd --mem 12G --num-threads 6 $score_cosine_snorm_dir/log/score_voxsrc22_dev.log \
	       hyperion-eval-verification-metrics \
	       --score-files $score_cosine_snorm_dir/voxsrc22_dev_scores.csv \
	       --key-files data/voxsrc22_dev/trials.csv \
	       --score-names voxsrc22_dev \
	       --key-names all \
	       --output-file $score_cosine_snorm_dir/voxsrc22_dev_results.csv

    cat $score_cosine_snorm_dir/voxsrc22_dev_results.csv

  fi

fi

if [ "$do_qmf" == "true" ];then
  if [ $stage -le 7 ];then
    echo "Train QMF in Vox2"
    echo "...Calculating quality measures for Vox2"
    num_parts=8
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do
	$train_cmd $score_cosine_qmf_dir/log/voxceleb2_trials_${i}_${j}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-cosine-scoring-backend-with-qmf \
		   --feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --ndx-file data/voxceleb2cat_train_trials/trials.csv \
		   --enroll-map-file data/voxceleb2cat_train_trials/enrollments.csv  \
		   --score-file $score_cosine_qmf_dir/voxceleb2_scores.csv \
		   --cohort-segments-file data/voxceleb2cat_train_cohort/segments.csv \
		   --cohort-feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --cohort-nbest 1000 --avg-cohort-by speaker \
		   --enroll-part-idx $i --num-enroll-parts $num_parts \
		   --test-part-idx $j --num-test-parts $num_parts &
      done
      sleep 5s
    done
    wait
    hyperion-merge-scores --output-file $score_cosine_qmf_dir/voxceleb2_scores.snorm.csv \
      			  --num-enroll-parts $num_parts --num-test-parts $num_parts

    hyperion-train-qmf --score-file $score_cosine_qmf_dir/voxceleb2_scores.snorm.csv \
		       --key-file data/voxceleb2cat_train_trials/trials.csv \
		       --model-file $score_cosine_qmf_dir/qmf.h5
    
  fi

  if [ $stage -le 8 ];then
    echo "Eval Voxceleb 1 with Cosine scoring + Adaptive SNorm + QMF"
    num_parts=16
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do
	$train_cmd --mem 22G $score_cosine_qmf_dir/log/voxceleb1_${i}_${j}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-cosine-scoring-backend-with-qmf \
		   --feats-file csv:$xvector_dir/voxceleb1_test/xvector.csv \
		   --ndx-file data/voxceleb1_test/trials.csv \
		   --enroll-map-file data/voxceleb1_test/enrollment.csv  \
		   --score-file $score_cosine_qmf_dir/voxceleb1_scores.csv \
		   --cohort-segments-file data/voxceleb2cat_train_cohort/segments.csv \
		   --cohort-feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --cohort-nbest 1000 --avg-cohort-by speaker \
		   --qmf-file $score_cosine_qmf_dir/qmf.h5 \
		   --enroll-part-idx $i --num-enroll-parts $num_parts \
		   --test-part-idx $j --num-test-parts $num_parts &
      done
      sleep 5s
    done
    wait
    for suffix in "" .snorm .snorm.qmf
    do
      (
	hyperion-merge-scores --output-file $score_cosine_qmf_dir/voxceleb1_scores$suffix.csv \
			      --num-enroll-parts $num_parts --num-test-parts $num_parts
	
	$train_cmd --mem 12G --num-threads 6 $score_cosine_qmf_dir/log/score_voxceleb1$suffix.log \
		   hyperion-eval-verification-metrics \
		   --score-files $score_cosine_qmf_dir/voxceleb1_scores$suffix.csv \
		   --key-files data/voxceleb1_test/trials_{o,e,h}.csv \
		   --score-names voxceleb1 \
		   --key-names O E H \
		   --sparse \
		   --output-file $score_cosine_qmf_dir/voxceleb1_results$suffix.csv

	echo "$score_cosine_qmf_dir/voxceleb1_results$suffix.csv:"
	cat $score_cosine_qmf_dir/voxceleb1_results$suffix.csv
      ) &
    done
    wait
  fi
  
  if [ $stage -le 9 ] && [ "$do_voxsrc22" == "true" ];then
    echo "Eval voxsrc2 with Cosine scoring + QMF"
    num_parts=16
    for((i=1;i<=$num_parts;i++));
    do
      for((j=1;j<=$num_parts;j++));
      do    
	$train_cmd $score_cosine_qmf_dir/log/voxsrc22_dev_${i}_${j}.log \
		   hyp_utils/conda_env.sh \
		   hyperion-eval-cosine-scoring-backend-with-qmf \
		   --feats-file csv:$xvector_dir/voxsrc22_dev/xvector.csv \
		   --ndx-file data/voxsrc22_dev/trials.csv \
		   --enroll-map-file data/voxsrc22_dev/enrollment.csv  \
		   --score-file $score_cosine_qmf_dir/voxsrc22_dev_scores.csv \
		   --cohort-segments-file data/voxceleb2cat_train_cohort/segments.csv \
		   --cohort-feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
		   --cohort-nbest 1000 --avg-cohort-by speaker \
		   --qmf-file $score_cosine_qmf_dir/qmf.h5 \
		   --enroll-part-idx $i --num-enroll-parts $num_parts \
		   --test-part-idx $j --num-test-parts $num_parts &
	sleep 5s
      done
      sleep 10s
    done
    wait
    for suffix in "" .snorm .snorm.qmf
    do
      (
	hyperion-merge-scores --output-file $score_cosine_qmf_dir/voxsrc22_dev_scores$suffix.csv \
			      --num-enroll-parts $num_parts --num-test-parts $num_parts

	$train_cmd --mem 12G --num-threads 6 $score_cosine_qmf_dir/log/score_voxsrc22_dev$suffix.log \
		   hyperion-eval-verification-metrics \
		   --score-files $score_cosine_qmf_dir/voxsrc22_dev_scores$suffix.csv \
		   --key-files data/voxsrc22_dev/trials.csv \
		   --score-names voxsrc22_dev \
		   --key-names all \
		   --output-file $score_cosine_qmf_dir/voxsrc22_dev_results$suffix.csv

	echo "$score_cosine_qmf_dir/voxsrc22_dev_results$suffix.csv:"
	cat $score_cosine_qmf_dir/voxsrc22_dev_results$suffix.csv
      ) &
    done
    wait
  fi

fi


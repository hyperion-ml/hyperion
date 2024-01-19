#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=2
nnet_stage=1
config_file=default_config.sh
use_gpu=false
xvec_chunk_length=120.0
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
fi

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name
score_cosine_dir=$score_dir/cosine
score_plda_dir=$score_dir/${cluster_name}_plda

if [ $stage -le 1 ]; then
  # Extract xvectors for training LDA/PLDA
  nj=100
  for name in voxceleb2cat_train
  do
    if [ -n "$vad_config" ];then
      vad_args="--vad csv:data/$name/vad.csv"
    fi
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} ${vad_args} \
	      --part-idx JOB --num-parts $nj  \
	      --recordings-file data/$name/recordings.csv \
	      --random-utt-length --min-utt-length 30 --max-utt-length 30 \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
fi

if [ $stage -le 2 ]; then
  # Extracts x-vectors for evaluation
  nj=100
  if [ "$do_voxsrc22" == "true" ];then
    extra_data="voxsrc22_dev"
  fi
  for name in voxceleb1_test $extra_data
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < 100 ? $num_segs:100))
    if [ -n "$vad_config" ];then
      vad_args="--vad csv:data/$name/vad.csv"
    fi
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} ${vad_args} \
	      --part-idx JOB --num-parts $nj  \
	      --recordings-file data/$name/recordings.csv \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
fi

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
  exit
fi


cluster_dir=exp/clustering/$nnet_s1_name/$cluster_name
if [ $stage -le 4 ];then
  echo "Cluster Vox2"
  mkdir -p $cluster_dir
  $train_cmd --mem 50G --num-threads 32 $cluster_dir/clustering.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV \
    hyperion-cluster-embeddings $cluster_method --cfg $cluster_cfg \
    --segments-file data/voxceleb2cat_train_xvector_train/segments.csv \
    --feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
    --output-file $cluster_dir/voxceleb2cat_train_xvector_train/segments.csv 
fi

if [ $stage -le 5 ];then
  echo "Train PLDA"
  $train_cmd $cluster_dir/plda.log \
	     hyp_utils/conda_env.sh --conda-env $HYP_ENV \
	     hyperion-train-plda --cfg $plda_cfg \
	     --segments-file $cluster_dir/voxceleb2cat_train_xvector_train/segments.csv \
	     --feats-file csv:$xvector_dir/voxceleb2cat_train/xvector.csv \
	     --preproc-file $cluster_dir/plda/preproc.h5 \
	     --plda-file $cluster_dir/plda/plda.h5

  
fi

if [ $stage -le 6 ];then

  echo "Eval Voxceleb 1 with PLDA"
  num_parts=8
  for((i=1;i<=$num_parts;i++));
  do
    for((j=1;j<=$num_parts;j++));
    do
      $train_cmd $score_plda_dir/log/voxceleb1_${i}_${j}.log \
		 hyp_utils/conda_env.sh \
		 hyperion-eval-plda-backend \
		 --feats-file csv:$xvector_dir/voxceleb1_test/xvector.csv \
		 --ndx-file data/voxceleb1_test/trials.csv \
		 --enroll-map-file data/voxceleb1_test/enrollment.csv  \
		 --score-file $score_plda_dir/voxceleb1_scores.csv \
		 --preproc-file $cluster_dir/plda/preproc.h5 \
		 --plda-file $cluster_dir/plda/plda.h5 \
		 --enroll-part-idx $i --num-enroll-parts $num_parts \
		 --test-part-idx $j --num-test-parts $num_parts &
    done
  done
  wait
  hyperion-merge-scores --output-file $score_plda_dir/voxceleb1_scores.csv \
			--num-enroll-parts $num_parts --num-test-parts $num_parts

  $train_cmd --mem 12G --num-threads 6 $score_plda_dir/log/score_voxceleb1.log \
	     hyperion-eval-verification-metrics \
	     --score-files $score_plda_dir/voxceleb1_scores.csv \
	     --key-files data/voxceleb1_test/trials_{o,e,h}.csv \
	     --score-names voxceleb1 \
	     --key-names O E H \
	     --sparse \
	     --output-file $score_plda_dir/voxceleb1_results.csv

  cat $score_plda_dir/voxceleb1_results.csv
  exit
fi
exit
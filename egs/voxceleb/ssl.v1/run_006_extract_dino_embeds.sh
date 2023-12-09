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

xvector_dir=exp/xvectors/$nnet_name
score_dir=exp/scores/$nnet_name
score_cosine_dir=$score_dir/cosine

if [[ $stage -le 1 && ( "$do_plda" == "true" || "$do_snorm" == "true" || "$do_qmf" == "true" || "$do_pca" == "true") ]]; then
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
	      --random-utt-length --min-utt-length 2 --max-utt-length 30 \
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
fi


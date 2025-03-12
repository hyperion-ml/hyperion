#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nnet_stage=""
config_file=default_config.sh
use_gpu=true
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

xvector_dir=exp/xvectors/$nnet_name

train_datasets="sre_cts_superset"


if [ $stage -le 1 ];then
  echo "Extract x-vector training data $train_datasets"
  nj=100
  for name in $train_datasets
  do
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} \
	      --part-idx JOB --num-parts $nj  \
	      --random-utt-length --min-utt-length 10. --max-utt-length 60. \
	      --recordings-file data/$name/recordings.csv \
	      --vad csv:data/$name/vad.csv \
	      --model-path $nnet \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    # for JOB in 10
    # do
    #   $xvec_cmd $output_dir/log/extract_xvectors.$JOB.log \
    # 		hyp_utils/conda_env.sh --num-gpus $num_gpus \
    # 		hyperion-extract-wav2xvectors ${xvec_args} \
    # 		--part-idx $JOB --num-parts $nj  \
    # 	        --random-utt-length --min-utt-length 10. --max-utt-length 60. \
    # 		--recordings-file data/$name/recordings.csv \
    # 		--vad csv:data/$name/vad.csv \
    # 		--model-path $nnet \
    # 		--output-spec ark,csv:$output_dir/xvector.$JOB.ark,$output_dir/xvector.$JOB.csv &
    # done
    # wait
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj
  done
fi

train_datasets="sre16_eval_train
sre21_audio_eval_enroll
sre21_audio_eval_test
sre21_audio-visual_eval_test"

if [ $stage -le 2 ];then
  echo "Extract x-vectors training data $train_datasets"
  nj=100
  for name in $train_datasets
  do
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} \
	      --part-idx JOB --num-parts $nj  \
	      --recordings-file data/$name/recordings.csv \
	      --vad csv:data/$name/vad.csv \
	      --model-path $nnet \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    # for JOB in 13
    # do
    #   $xvec_cmd $output_dir/log/extract_xvectors.$JOB.log \
    # 		hyp_utils/conda_env.sh --num-gpus $num_gpus \
    # 		hyperion-extract-wav2xvectors ${xvec_args} \
    # 		--part-idx $JOB --num-parts $nj  \
    # 		--recordings-file data/$name/recordings.csv \
    # 		--vad csv:data/$name/vad.csv \
    # 		--model-path $nnet \
    # 		--output-spec ark,csv:$output_dir/xvector.$JOB.ark,$output_dir/xvector.$JOB.csv &
    # done
    # wait
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj
  done
fi

if [ $stage -le 3 ]; then
  echo "Extracts x-vectors for evaluation"
  max_jobs=100
  for name in sre21_audio_dev_enroll sre21_audio_dev_test sre21_audio-visual_dev_test sre24_audio_dev_test sre24_audio-visual_dev_test sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $max_jobs ? $num_segs:$max_jobs))
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} \
	      --part-idx JOB --num-parts $nj  \
	      --vad csv:data/$name/vad.csv \
	      --recordings-file data/$name/recordings.csv \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
fi

if [ $stage -le 4 ];then
  echo "Extracts x-vectors for multi-speaker enrollment datasets"
  max_jobs=20
  for name in sre24_audio_dev_enroll sre24_audio_eval_enroll
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $max_jobs ? $num_segs:$max_jobs))
    output_dir=$xvector_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-extract-wav2xvectors ${xvec_args} ${vad_args} \
	      --part-idx JOB --num-parts $nj  \
	      --recordings-file data/$name/recordings.csv \
	      --vad csv:data/$name/vad_mixed.csv \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
  
fi

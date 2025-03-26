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
win_length=3.0
win_shift=1.0
ahc_threshold=0.0
min_cluster_duration=2.0
ahc_max_clusters=4

. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_diar_chunk_length"
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
elif [ $nnet_stage -eq 4 ];then
  nnet=$nnet_s4
  nnet_name=$nnet_s4_name
fi

if [[ $nnet_type =~ ^hf_ ]]; then
  extract_bin=local/extract_wav2vec2xvectors_diarization.py
else
  extract_bin=local/extract_wav2xvectors_diarization.py
fi

xvector_dir=exp/xvectors/${nnet_name}
xvector_diar_dir=exp/xvectors/${nnet_name}/$diar_label
be_dir=exp/be/$nnet_name
be_sre24_dir=$be_dir/$be_sre24_name
score_dir=exp/scores/$nnet_name
score_plda_dir=$score_dir/${be_sre24_name}/plda
score_plda_cal_dir=${score_plda_dir}_cal_v2_folds


if [ $stage -le 1 ]; then
  echo "Extracts x-vectors for evaluation"
  max_jobs=100
  for name in sre24_audio_dev_test sre24_audio-visual_dev_test sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $max_jobs ? $num_segs:$max_jobs))
    output_dir=$xvector_diar_dir/$name
    echo "Extracting x-vectors for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      $extract_bin ${xvec_args} \
	      --cfg $diar_sre24_cfg \
	      --part-idx JOB --num-parts $nj  \
	      --vad csv:data/$name/vad.csv \
	      --recordings-file data/$name/recordings.csv \
	      --segments-file data/$name/segments.csv \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv \
	      --debug-dir $output_dir/score_hist \
	      --preproc-file $be_sre24_dir/preproc_adapt.pkl \
	      --plda-file $be_sre24_dir/plda_adapt.h5 \
	      --calibration-file ${score_plda_cal_dir}/calibration.h5 \
    
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
fi

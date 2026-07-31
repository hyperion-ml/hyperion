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
cuda_eval_cmd=$cuda_cmd
if [ "$use_gpu" == "true" ];then
  xvec_args="--use-gpu --chunk-length $xvec_chunk_length"
  xvec_cmd="$cuda_eval_cmd --gpu 1 --mem 6G"
  num_gpus=1
else
  xvec_cmd="$train_cmd --mem 12G"
  num_gpus=0
fi

nnet=$lid_nnet_s1
nnet_name=$lid_nnet_s1_name

xvector_dir=exp/lid_xvectors/$nnet_name
be_name=pca_cw_lnorm_lgbe
be_dir=exp/lid_be/$nnet_name/$be_name
score_dir=exp/lid_scores/$nnet_name/$be_name

if [ $stage -le 1 ]; then
  echo "Extracts LID x-vectors"
  max_jobs=100
  for name in sre24_audio_dev_enroll sre24_audio_dev_test sre24_audio-visual_dev_test \
				     sre24_audio_eval_enroll sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $max_jobs ? $num_segs:$max_jobs))
    output_dir=$xvector_dir/$name
    echo "Extracting LID x-vectors for $name"
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



if [ $stage -le 2 ];then
  echo "Train Gaussian Back-end to recognize Arabic, English and French"
  $train_cmd $be_dir/train_lgbe.log \
	     hyp_utils/conda_env.sh \
	     hyperion-train-lgbe \
	     --segments-files data/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/segments.csv \
	     --feats-files $xvector_dir/sre24_audio{_dev_enroll,_dev_test,-visual_dev_test}/xvector.csv \
	     --preproc-file $be_dir/preproc.h5 \
	     --lgbe-file $be_dir/lgbe.h5 \
	     --pca.pca-var-r 1.0 --lgbe-lnorm --lgbe-center --lgbe-whiten
fi

if [ $stage -le 3 ];then
  echo "Eval LID scores"
  for name in sre24_audio_eval_enroll sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    echo "Eval LID scores for $name in $score_dir"
    $train_cmd $score_dir/log/eval_lid_${name}.log \
	       hyp_utils/conda_env.sh \
	       hyperion-eval-lgbe \
	       --segments-file data/$name/segments.csv \
	       --feats-file $xvector_dir/$name/xvector.csv \
	       --preproc-file $be_dir/preproc.h5 \
	       --lgbe-file $be_dir/lgbe.h5 \
	       --score-file $score_dir/${name}_scores.csv &
	       
  done
  wait
fi

if [ $stage -le 4 ];then
  echo "Add LID info to data dirs"
  for name in sre24_audio_eval_enroll sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    echo "Add LID info to segments in $name"
    python local/add_lang_to_segments.py \
	   --segments-file data/$name/segments.csv \
	   --score-file $score_dir/${name}_scores.csv
  done

  for name in sre24_audio_eval_test sre24_audio-visual_eval_test
  do
    echo "Add LID info to trials in $name"
    python local/add_lang_to_trials.py \
	   --enroll-map-file data/sre24_audio_eval_enroll/enrollment.csv \
	   --enroll-segments-file data/sre24_audio_eval_enroll/segments.csv \
	   --test-segments-file data/$name/segments.csv \
	   --ndx-file data/$name/trials.tsv
    cp data/$name/trials.tsv data/$name/trials_ext.tsv
  done
  
fi

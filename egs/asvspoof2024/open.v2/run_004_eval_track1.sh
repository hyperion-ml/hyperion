#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
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

logit_dir=exp/cm_scores/$nnet_name

# if [[ $stage -le 1 && ( "$do_plda" == "true" || "$do_snorm" == "true" || "$do_qmf" == "true" || "$do_pca" == "true") ]]; then
#   # Extract xvectors for training LDA/PLDA
#   nj=100
#   for name in voxceleb2cat_train
#   do
#     if [ -n "$vad_config" ];then
#       vad_args="--vad csv:data/$name/vad.csv"
#     fi
#     output_dir=$logit_dir/$name
#     echo "Extracting x-vectors for $name"
#     $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
# 	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
# 	      hyperion-eval-wav2xvectors-logits ${xvec_args} ${vad_args} \
# 	      --part-idx JOB --num-parts $nj  \
# 	      --recordings-file data/$name/recordings.csv \
# 	      --random-utt-length --min-utt-length 2 --max-utt-length 30 \
# 	      --model-path $nnet  \
# 	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
#     hyperion-tables cat \
# 		    --table-type features \
# 		    --output-file $output_dir/xvector.csv --num-tables $nj

#   done
# fi

if [ $stage -le 2 ]; then
  nj=10
  for name in asvspoof2024_dev asvspoof2024_prog
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $nj ? $num_segs:$nj))
    if [ -n "$vad_config" ];then
      vad_args="--vad csv:data/$name/vad.csv"
    fi
    output_dir=$logit_dir/$name
    echo "Extracting logits for $name"
    $xvec_cmd JOB=1:$nj $output_dir/log/extract_logits.JOB.log \
	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
	      hyperion-eval-wav2vec2xvector-logits ${xvec_args} ${vad_args} \
	      --part-idx JOB --num-parts $nj  \
	      --recordings-file data/$name/recordings.csv \
	      --model-path $nnet  \
	      --output-spec ark,csv:$output_dir/logits.JOB.ark,$output_dir/logits.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/logits.csv --num-tables $nj
    python local/spoof_logits_to_trial_scores.py \
	   --segments-file data/$name/segments.csv \
	   --logits-spec csv:$output_dir/logits.csv \
	   --score-file $output_dir/scores.tsv
  done
fi

prior=0.655
if [ $stage -le 3 ];then
  # ASVSpoof2024 p_spoof=0.05, c_miss=1, c_fa=10
  # Target class is bonafide so p_tar=1-p_spoof=0.95
  # effective prior for calibration
  # prior = effective_prior(0.95, c_mis=1, c_fa=10)=0.655
  
  hyperion-train-verification-calibration \
    --key-files data/asvspoof2024_dev/trials_track1.csv \
    --score-files $logit_dir/asvspoof2024_dev/scores.tsv \
    --prior ${prior} \
    --model-file $logit_dir/asvspoof2024_dev/cal_p${prior}.h5

  hyperion-eval-verification-calibration \
    --ndx-file data/asvspoof2024_dev/trials_track1.csv \
    --in-score-file $logit_dir/asvspoof2024_dev/scores.tsv \
    --out-score-file $logit_dir/asvspoof2024_dev/scores_cal_p${prior}.tsv \
    --model-file $logit_dir/asvspoof2024_dev/cal_p${prior}.h5

  hyperion-eval-verification-calibration \
    --ndx-file data/asvspoof2024_prog/trials_track1.csv \
    --in-score-file $logit_dir/asvspoof2024_prog/scores.tsv \
    --out-score-file $logit_dir/asvspoof2024_prog/scores_cal_p${prior}.tsv \
    --model-file $logit_dir/asvspoof2024_dev/cal_p${prior}.h5

  hyperion-eval-verification-metrics \
    --key-files data/asvspoof2024_dev/trials_track1.csv \
    --score-files $logit_dir/asvspoof2024_dev/scores_cal_p${prior}.tsv \
    --key-names all \
    --score-names all \
    --p-tar 0.95 --c-miss 1 --c-fa 10 \
    --output-file $logit_dir/asvspoof2024_dev/results.tsv
fi

if [ $stage -le 4 ];then
  echo "convert to asvspoof5 format and eval with official tool"
  python local/cm_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_dev/trials_track1.csv \
	 --in-score-file $logit_dir/asvspoof2024_dev/scores_cal_p${prior}.tsv \
	 --out-score-file $logit_dir/asvspoof2024_dev/official/score.tsv
  python local/cm_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_prog/trials_track1.csv \
	 --in-score-file $logit_dir/asvspoof2024_prog/scores_cal_p${prior}.tsv \
	 --out-score-file $logit_dir/asvspoof2024_prog/official/score.tsv

  python asvspoof5/evaluation-package/evaluation.py \
	 --m t1 \
	 --cm $logit_dir/asvspoof2024_dev/official/score.tsv \
	 --cm_key data/asvspoof2024_dev/trials_track1_official.tsv \
	 | tee $logit_dir/asvspoof2024_dev/official/results.txt
	 

fi

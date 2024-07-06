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

cm_score_dir=exp/cm_scores/$nnet_name
xvector_dir=exp/xvectors/$spk_nnet_name
asv_score_dir=exp/asv_scores/$spk_nnet_name
asv_score_cosine_dir=$asv_score_dir/cosine
asv_score_cosine_cal_dir=$asv_score_dir/cosine_cal
asv_score_cosine_cal_dir=$asv_score_dir/cosine_cal
asvspoof_score_dir=exp/asvspoof_scores/${spk_nnet_name}_cosine/$nnet_name


# if [[ $stage -le 1 && ( "$do_plda" == "true" || "$do_snorm" == "true" || "$do_qmf" == "true" || "$do_pca" == "true") ]]; then
#   # Extract xvectors for training LDA/PLDA
#   nj=100
#   for name in voxceleb2cat_train
#   do
#     if [ -n "$vad_config" ];then
#       vad_args="--vad csv:data/$name/vad.csv"
#     fi
#     output_dir=$xvector_dir/$name
#     echo "Extracting x-vectors for $name"
#     $xvec_cmd JOB=1:$nj $output_dir/log/extract_xvectors.JOB.log \
# 	      hyp_utils/conda_env.sh --num-gpus $num_gpus \
# 	      hyperion-extract-wav2xvectors ${xvec_args} ${vad_args} \
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
  # Extracts x-vectors for evaluation
  nj=10
  for name in asvspoof2024_dev_enroll asvspoof2024_dev asvspoof2024_prog_enroll asvspoof2024_prog
  do
    num_segs=$(wc -l data/$name/segments.csv | awk '{ print $1-1}')
    nj=$(($num_segs < $nj ? $num_segs:$nj))
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
	      --model-path $spk_nnet  \
	      --output-spec ark,csv:$output_dir/xvector.JOB.ark,$output_dir/xvector.JOB.csv
    hyperion-tables cat \
		    --table-type features \
		    --output-file $output_dir/xvector.csv --num-tables $nj

  done
fi

if [ $stage -le 3 ];then
  echo "Evaluate SV"
  $train_cmd $asv_score_cosine_dir/log/asvspoof2024_dev.log \
	     hyp_utils/conda_env.sh \
	     hyperion-eval-cosine-scoring-backend \
	     --enroll-feats-file csv:$xvector_dir/asvspoof2024_dev_enroll/xvector.csv \
	     --feats-file csv:$xvector_dir/asvspoof2024_dev/xvector.csv \
	     --ndx-file data/asvspoof2024_dev/trials_track2.csv \
	     --enroll-map-file data/asvspoof2024_dev_enroll/enroll.csv \
	     --score-file $asv_score_cosine_dir/asvspoof2024_dev_scores.csv
  
  $train_cmd $asv_score_cosine_dir/log/asvspoof2024_prog.log \
	     hyp_utils/conda_env.sh \
	     hyperion-eval-cosine-scoring-backend \
	     --enroll-feats-file csv:$xvector_dir/asvspoof2024_prog_enroll/xvector.csv \
	     --feats-file csv:$xvector_dir/asvspoof2024_prog/xvector.csv \
	     --ndx-file data/asvspoof2024_prog/trials_track2.csv \
	     --enroll-map-file data/asvspoof2024_prog_enroll/enroll.csv \
	     --score-file $asv_score_cosine_dir/asvspoof2024_prog_scores.csv

fi

if [ $stage -le 4 ];then
  echo "Calibrate ASV Scores"
  # ASVSpoof 5
  # priors -> ptar=0.9405 pnon=0.0095 pspoof=0.05
  # costs -> c_miss=1, c_fa=10, c_fa_spoof=10
  # We calibrate SV on non-spoof trials
  # We need to get priors|nonspoof -> ptar|nonspoof=0.9405/(0.9405+0095)=0.99
  # The effective prior | nonspoof = effective_prior(0.99, c_fa=10, c_miss=1)=0.90
  # We calibrate using the effecitve prior | nonspoof
  hyperion-train-verification-calibration \
    --key-file data/asvspoof2024_dev/trials_track2.csv \
    --score-file $asv_score_cosine_dir/asvspoof2024_dev_scores.csv \
    --prior 0.90 \
    --model-file $asv_score_cosine_cal_dir/cal.h5

  hyperion-eval-verification-calibration \
    --ndx-file data/asvspoof2024_dev/trials_track2.csv \
    --in-score-file $asv_score_cosine_dir/asvspoof2024_dev_scores.csv \
    --out-score-file $asv_score_cosine_cal_dir/asvspoof2024_dev_scores.csv  \
    --model-file $asv_score_cosine_cal_dir/cal.h5

  hyperion-eval-verification-calibration \
    --ndx-file data/asvspoof2024_prog/trials_track2.csv \
    --in-score-file $asv_score_cosine_dir/asvspoof2024_prog_scores.csv \
    --out-score-file $asv_score_cosine_cal_dir/asvspoof2024_prog_scores.csv  \
    --model-file $asv_score_cosine_cal_dir/cal.h5

  hyperion-eval-verification-metrics \
    --key-files data/asvspoof2024_dev/trials_track2.csv \
    --score-files $asv_score_cosine_cal_dir/asvspoof2024_dev_scores.csv \
    --key-names all \
    --score-names all \
    --p-tar 0.99 --c-miss 1 --c-fa 10 \
    --output-file $asv_score_cosine_cal_dir/asvspoof2024_dev_results.tsv
fi

if [ $stage -le 5 ];then
  echo "Combine with spoofing scores"
  python local/combine_asv_cm_scores.py \
	 --asv-score-file $asv_score_cosine_cal_dir/asvspoof2024_dev_scores.csv \
	 --cm-score-file $cm_score_dir/asvspoof2024_dev/scores_cal_p0.655.tsv \
	 --out-score-file $asvspoof_score_dir/asvspoof2024_dev_scores.tsv \
	 --p-tar 0.9405 --p-spoof 0.05 --c-miss 1 --c-fa 10 --c-fa-spoof 10
  python local/combine_asv_cm_scores.py \
	 --asv-score-file $asv_score_cosine_cal_dir/asvspoof2024_prog_scores.csv \
	 --cm-score-file $cm_score_dir/asvspoof2024_prog/scores_cal_p0.655.tsv \
	 --out-score-file $asvspoof_score_dir/asvspoof2024_prog_scores.tsv \
	 --p-tar 0.9405 --p-spoof 0.05 --c-miss 1 --c-fa 10 --c-fa-spoof 10
  
fi

if [ $stage -le 6 ];then
  echo "convert to asvspoof5 format and eval with official tool"
  python local/asvspoof_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_dev/trials_track2.csv \
	 --asv-score-file $asv_score_cosine_cal_dir/asvspoof2024_dev_scores.csv \
	 --cm-score-file $cm_score_dir/asvspoof2024_dev/scores_cal_p0.655.tsv \
	 --asvspoof-score-file $asvspoof_score_dir/asvspoof2024_dev_scores.tsv \
	 --out-score-file $asvspoof_score_dir/official/asvspoof2024_dev/score.tsv
  python local/asvspoof_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_prog/trials_track2.csv \
	 --asv-score-file $asv_score_cosine_cal_dir/asvspoof2024_prog_scores.csv \
	 --cm-score-file $cm_score_dir/asvspoof2024_prog/scores_cal_p0.655.tsv \
	 --asvspoof-score-file $asvspoof_score_dir/asvspoof2024_prog_scores.tsv \
	 --out-score-file $asvspoof_score_dir/official/asvspoof2024_prog/score.tsv

  python asvspoof5/evaluation-package/evaluation.py \
	 --m t2_tandem \
	 --sasv $asvspoof_score_dir/official/asvspoof2024_dev/score.tsv \
	 --sasv_key data/asvspoof2024_dev/trials_track2_official.tsv \
	 | tee $asvspoof_score_dir/official/asvspoof2024_dev/results.txt

fi

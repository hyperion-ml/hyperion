#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
prior=0.655
#nnet_stage=1
#config_file=default_config.sh
. parse_options.sh || exit 1;
#. $config_file

# system_names="lresnet34_v1.1 USC_S4_v1"
# system_dirs="exp/cm_scores/fbank80_stmn_lresnet34.v1.1_a100.s1
# exp/cm_scores/USC_S4_v1"
# output_dir=exp/cm_scores/fusion_v1
# fus_idx=1

# system_names="lresnet34_v1.1 USC_S4_v1 USC_S4_v1_rir_ep10"
# system_dirs="exp/cm_scores/fbank80_stmn_lresnet34.v1.1_a100.s1
# exp/cm_scores/USC_S4_v1
# exp/cm_scores/USC_S4_v1_rir_ep10"
# output_dir=exp/cm_scores/fusion_v2
# fus_idx=2

system_names="lresnet34_v1.17e USC_S4_v1"
system_dirs="exp/cm_scores/fbank80_stmn_lresnet34.v1.17e_a100.s1
exp/cm_scores/USC_S4_v1"
output_dir=exp/cm_scores/fusion_v3
fus_idx=1

score_files_dev=""
score_files_prog=""
for d in $system_dirs
do
  score_files_dev="$score_files_dev $d/asvspoof2024_dev/scores.tsv"
  score_files_prog="$score_files_prog $d/asvspoof2024_prog/scores.tsv"
done

if [ $stage -le 1 ];then
  hyperion-train-verification-greedy-fusion \
    --key-file data/asvspoof2024_dev/trials_track1.csv \
    --system-names $system_names \
    --score-files $score_files_dev \
    --model-file $output_dir/fus.h5 \
    --prior $prior --lambda-reg 1e-3 

  hyperion-eval-verification-greedy-fusion \
    --ndx-file data/asvspoof2024_dev/trials_track1.csv \
    --in-score-files $score_files_dev \
    --out-score-file $output_dir/asvspoof2024_dev/scores.tsv \
    --model-file $output_dir/fus.h5 --fus-idx $fus_idx
    
  hyperion-eval-verification-greedy-fusion \
    --ndx-file data/asvspoof2024_prog/trials_track1.csv \
    --in-score-files $score_files_prog \
    --out-score-file $output_dir/asvspoof2024_prog/scores.tsv \
    --model-file $output_dir/fus.h5 --fus-idx $fus_idx

    hyperion-eval-verification-metrics \
    --key-files data/asvspoof2024_dev/trials_track1.csv \
    --score-files $output_dir/asvspoof2024_dev/scores.tsv \
    --key-names all \
    --score-names all \
    --p-tar 0.95 --c-miss 1 --c-fa 10 \
    --output-file $output_dir/asvspoof2024_dev/results.tsv

fi

if [ $stage -le 2 ];then
  echo "convert to asvspoof5 format and eval with official tool"
  python local/cm_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_dev/trials_track1.csv \
	 --in-score-file $output_dir/asvspoof2024_dev/scores.tsv \
	 --out-score-file $output_dir/asvspoof2024_dev/official/score.tsv
  python local/cm_scores_to_asvspoof5_format.py \
	 --ndx-file data/asvspoof2024_prog/trials_track1.csv \
	 --in-score-file $output_dir/asvspoof2024_prog/scores.tsv \
	 --out-score-file $output_dir/asvspoof2024_prog/official/score.tsv

  python asvspoof5/evaluation-package/evaluation.py \
	 --m t1 \
	 --cm $output_dir/asvspoof2024_dev/official/score.tsv \
	 --cm_key data/asvspoof2024_dev/trials_track1_official.tsv \
	 | tee $output_dir/asvspoof2024_dev/official/results.txt

fi


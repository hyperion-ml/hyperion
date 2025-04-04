#!/bin/bash

echo "| Model | SRE24 Dev Folds | | | SRE24 Dev Full (Cheat) | | | SRE24 Eval | | |"
echo "| ----  | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |"
echo "| | EER(%) | Min Cp | Act Cp | EER(%) | Min Cp | Max Cp | EER(%) | Min Cp | Max Cp |"
for name in fbank64_stmn_nb_fwseres2net50w26s8.v3.2.s2  fbank64_stmn_nb_idrnd_resnet100.v3.1.s2  fbank80_stmn_fwseres2net50w26s8.v3.1.s2  fbank80_stmn_idrnd_resnet100.v3.1.s2
do
  dir=exp/scores/$name/plda_adapt_sre24/plda_diarization_cal_v2_folds
  awk '/equal/ { printf "| '$name' | %.2f | %.3f | %.3f | ",$4,$9,$10}' $dir/folds/0+1/sre24_audio_dev_results.tsv
  awk '/equal/ { printf "%.2f | %.3f | %.3f | ",$4,$9,$10}' $dir/sre24_audio_dev_results.tsv
  awk '/equal/ { printf "%.2f | %.3f | %.3f |\n",$4,$9,$10}' $dir/sre24_audio_eval_results.tsv
done

for name in v1_pfus0.01_l21e-3
do
  dir=exp/fusion/$name/3
  awk '/equal/ { printf "| '$name' | %.2f | %.3f | %.3f | ",$4,$9,$10}' $dir/folds/0+1/sre24_audio_dev_results.tsv
  awk '/equal/ { printf "%.2f | %.3f | %.3f | ",$4,$9,$10}' $dir/sre24_audio_dev_results.tsv
  awk '/equal/ { printf "%.2f | %.3f | %.3f | \n",$4,$9,$10}' $dir/sre24_audio_eval_results.tsv
done

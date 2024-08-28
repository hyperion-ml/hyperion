#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. datapath.sh 
. $config_file


if [ $stage -le 1 ];then
  echo "Prepare the ASV Spoof 2024 train dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset train \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_train

  echo "Prepare the ASV Spoof 2024 dev-enroll dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset dev_enroll \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_dev_enroll

  echo "Prepare the ASV Spoof 2024 dev-test dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset dev \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_dev
  
fi

if [ $stage -le 2 ];then
  echo "Prepare the ASV Spoof 2024 progress dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset progress \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_prog

  echo "Prepare the ASV Spoof 2024 progress-enroll dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset progress_enroll \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_prog_enroll

  
fi

if [ $stage -le 3 ];then
  # echo "Prepare the ASV Spoof 2024 eval dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset eval \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_eval

  echo "Prepare the ASV Spoof 2024 eval-enroll dataset"
  hyperion-prepare-data asvspoof2024 \
			--subset eval_enroll \
			--corpus-dir $asvspoof2024_root \
			--output-dir data/asvspoof2024_eval_enroll

  
fi
exit

if [ $stage -le 4 ];then
  if [ ! -d ./asvspoof5 ];then
    git clone https://github.com/asvspoof-challenge/asvspoof5.git
  fi
  awk '
BEGIN{
  FS=","; OFS="\t"; 
  getline; 
  print "filename\tcm-label"
} 
{ 
  sub("nontarget","spoof", $3); sub("target","bonafide", $3);
  print $2,$3;
}' data/asvspoof2024_dev/trials_track1.csv > \
      data/asvspoof2024_dev/trials_track1_official.tsv
  
  awk '
BEGIN{
  FS=","; OFS="\t"; 
  getline; 
  print "spk\tfilename\tcm-label\tasv-label"
} 
{ 
  if($3 == "spoof") { cm="spoof"} else {cm="bonafide"};
  print $1,$2,cm,$3;
}' data/asvspoof2024_dev/trials_track2.csv > \
      data/asvspoof2024_dev/trials_track2_official.tsv 

fi


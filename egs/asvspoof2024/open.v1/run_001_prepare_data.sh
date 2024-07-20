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
  # Prepare the ASV Spoof 2015 dataset for training.
  hyperion-prepare-data asvspoof2015 \
  			--subset train \
  			--corpus-dir $asvspoof2015_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2015_train
  hyperion-prepare-data asvspoof2015 \
  			--subset dev \
  			--corpus-dir $asvspoof2015_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2015_dev
  hyperion-prepare-data asvspoof2015 \
			--subset eval \
			--corpus-dir $asvspoof2015_root \
			--use-kaldi-ids \
			--output-dir data/asvspoof2015_eval

fi

if [ $stage -le 2 ];then
  # Prepare the ASV Spoof 2017 dataset for training.
  hyperion-prepare-data asvspoof2017 \
  			--subset train \
  			--corpus-dir $asvspoof2017_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2017_train
  hyperion-prepare-data asvspoof2017 \
  			--subset dev \
  			--corpus-dir $asvspoof2017_root \
  			--use-kaldi-ids \
  			--output-dir data/asvspoof2017_dev
  hyperion-prepare-data asvspoof2017 \
			--subset eval \
			--corpus-dir $asvspoof2017_root \
			--use-kaldi-ids \
			--output-dir data/asvspoof2017_eval

fi

if [ $stage -le 3 ];then
  echo "Prepare the ASV Spoof 2019 LA dataset"
  hyperion-prepare-data asvspoof2019 \
			--subset la_train \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_train
  hyperion-prepare-data asvspoof2019 \
			--subset la_dev \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_dev
  hyperion-prepare-data asvspoof2019 \
			--subset la_eval \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_eval
  hyperion-prepare-data asvspoof2019 \
			--subset la_dev_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_dev_enroll
  hyperion-prepare-data asvspoof2019 \
			--subset la_eval_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_la_eval_enroll

  echo "Prepare the ASV Spoof 2019 PA dataset"
  hyperion-prepare-data asvspoof2019 \
			--subset pa_train \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_train
  hyperion-prepare-data asvspoof2019 \
			--subset pa_dev \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_dev
  hyperion-prepare-data asvspoof2019 \
			--subset pa_eval \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_eval
  hyperion-prepare-data asvspoof2019 \
			--subset pa_dev_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_dev_enroll
  hyperion-prepare-data asvspoof2019 \
			--subset pa_eval_enroll \
			--corpus-dir $asvspoof2019_root \
			--output-dir data/asvspoof2019_pa_eval_enroll

fi

if [ $stage -le 4 ];then
  echo "Prepare the ASV Spoof 2021 LA dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset la_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_la_eval
  
  echo "Prepare the ASV Spoof 2021 DF dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset df_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_df_eval

  echo "Prepare the ASV Spoof 2021 PA dataset"
  hyperion-prepare-data asvspoof2021 \
			--subset pa_eval \
			--corpus-dir $asvspoof2021_root \
			--output-dir data/asvspoof2021_pa_eval

fi

if [ $stage -le 5 ];then
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

if [ $stage -le 6 ];then
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

if [ $stage -le 7 ];then
  if [ ! -d ./asvspoof5 ];then
    if [ -d ../closed.v1/asvspoof5 ];then
      ln -s ../closed.v1/asvspoof5
    else
      git clone https://github.com/asvspoof-challenge/asvspoof5.git
    fi
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
  print "filename\tcm-label\tasv-label"
} 
{ 
  if($3 == "spoof") { cm="spoof"} else {cm="bonafide"};
  print $2,cm,$3;
}' data/asvspoof2024_dev/trials_track2.csv > \
      data/asvspoof2024_dev/trials_track2_official.tsv 

fi

if [ $stage -le 8 ];then
  hyperion-dataset merge \
		   --dataset data/train_open_la \
		   --input-datasets data/asvspoof2015_{train,dev,eval} \
		   data/asvspoof2019_la_{train,dev,dev_enroll,eval,eval_enroll} \
		   data/asvspoof2021_{df,la}_eval \
		   data/asvspoof2024_train
fi


if [ $stage -le 9 ];then
  echo "Prepare lhotse LibriSpeech manifest"
  # We assume that you have downloaded the LibriSpeech corpus
  # to $librispeech_root
  mkdir -p data/lhotse_librispeech
  if [ ! -e data/lhotse_librispeech/.librispeech.done ]; then
    nj=6
    lhotse prepare librispeech -j $nj $librispeech_root data/lhotse_librispeech
    touch data/lhotse_librispeech/.librispeech.done
  fi
  echo "Convert Manifest to Hyperion Datasets"
  for data in train-clean-100 train-clean-360 train-other-500
  do
    hyperion-dataset from_lhotse \
		     --recordings-file data/lhotse_librispeech/librispeech_recordings_${data}.jsonl.gz \
		     --supervisions-file data/lhotse_librispeech/librispeech_supervisions_${data}.jsonl.gz \
		     --dataset data/librispeech_${data}
  done
  echo "Merge Librispeech train sets"
  hyperion-dataset merge \
		   --input-datasets data/librispeech_train-{clean-100,clean-360,other-500} \
		   --dataset data/librispeech_train-960
fi

if [ $stage -le 10 ];then
  echo "Prepare the VoxCeleb2 dataset for training."
  hyperion-prepare-data voxceleb2 --subset dev --corpus-dir $voxceleb2_root \
			--cat-videos --use-kaldi-ids \
			--output-dir data/voxceleb2cat_train
fi

if [ $stage -le 11 ];then
  for data in voxceleb2cat_train
  do
    echo "Convert $data to flac"
    nj=40
    output_dir=exp/proc_audio/$data
    $train_cmd JOB=1:$nj $output_dir/log/preproc_audios_${data}.JOB.log \
	       hyp_utils/conda_env.sh \
	       hyperion-preprocess-audio-files \
	       --audio-format flac --remove-dc-offset \
	       --part-idx JOB --num-parts $nj \
	       --recordings-file data/$data/recordings.csv \
	       --output-path $output_dir \
	       --output-recordings-file $output_dir/recordings.JOB.csv

    hyperion-tables cat \
		    --table-type recordings \
		    --output-file $output_dir/recordings.csv --num-tables $nj
    
    hyperion-dataset set_recordings \
		     --dataset data/$data \
		     --recordings-file $output_dir/recordings.csv \
		     --output-dataset data/${data}_proc_audio 
  done
fi

if [ $stage -le 12 ];then
  for data in voxceleb2cat_train_proc_audio librispeech_train-960
  do
    echo "Add spoof_det column to $data"
    cp data/$data/segments.csv data/$data/segments.csv.bk
    awk 'BEGIN{ FS=","; OFS=","} {if(NR==1){ print $0,"spoof_det"}else{ print $0,"bonafide" }}' \
	data/$data/segments.csv.bk > data/$data/segments.csv
  done
fi

if [ $stage -le 13 ];then
  echo "Merge ASVSpoof data with Libri and Vox2"
  hyperion-dataset merge \
		   --dataset data/train_open_la_libri \
		   --input-datasets data/train_open_la data/librispeech_train-960
  hyperion-dataset merge \
		   --dataset data/train_open_la_libri_vox2 \
		   --input-datasets data/train_open_la_libri data/voxceleb2cat_train_proc_audio

fi

if [ $stage -le 14 ];then
  echo "prepare fake codec data"
  hyperion-prepare-data fake_codec \
			--hf-data-path $codec_fake_hf_root \
			--corpus-dir $codec_fake_root \
			--output-dir data/codec_fake
  
fi

if [ $stage -le 15 ];then
  echo "Merge ASVSpoof data with Codec Fake"
  hyperion-dataset merge \
		   --dataset data/train_open_la_cf \
		   --input-datasets data/train_open_la data/codec_fake
  
fi

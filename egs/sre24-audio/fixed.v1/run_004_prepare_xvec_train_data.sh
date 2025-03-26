#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

nodes=fs06
nj=40
stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

datasets="sre_cts_superset
sre16_eval_train
sre21_audio-visual_eval_test
sre21_audio_eval_enroll
sre21_audio_eval_test"

if [ $stage -le 1 ]; then
  # Prepare to distribute data over multiple machines
  # This only does something at CLSP grid
  for name in $datasets
  do
    hyp_utils/create_data_split_dirs.sh \
      exp/xvector_audios/$name \
      $USER/hyp-data/sre24-audio/xvector_audios/$name $nodes
  done
fi

if [ $stage -le 2 ];then
  for name in $datasets
  do
    echo "Processing $name for training"
    output_dir=exp/proc_audio/$name
    # This creates links to distribute data in CLSP grid
    # If you are not at CLSP grid, it does nothing and can be deleted
    hyp_utils/create_audios_split_links.sh $output_dir data/$name/recordings.csv flac
    if [ -n "$vad_config" ];then
      vad_args="--vad csv:data/$name/vad.csv"
      update_durs="--update-seg-durs"
    fi

    $train_cmd JOB=1:$nj $output_dir/log/preproc_audios_${name}.JOB.log \
	       hyp_utils/conda_env.sh \
	       hyperion-preprocess-audio-files \
	       --audio-format flac --remove-dc-offset $vad_args \
	       --part-idx JOB --num-parts $nj \
	       --recordings-file data/$name/recordings.csv \
	       --output-path $output_dir \
	       --output-recordings-file $output_dir/recordings.JOB.csv

    hyperion-tables cat \
		    --table-type recordings \
		    --output-file $output_dir/recordings.csv --num-tables $nj
    
    hyperion-dataset set_recordings $update_durs \
		     --dataset data/$name \
		     --recordings-file $output_dir/recordings.csv \
		     --output-dataset data/${name}_proc_audio \
		     --remove-vads vad
  done
fi

if [ $stage -le 3 ];then
  echo "Mergin training datasets"
  hyperion-dataset merge \
                   --dataset data/sre96-12_16_21_proc_audio \
                   --input-datasets data/{sre_cts_superset,sre16_eval_train}_proc_audio \
		   data/sre21_audio_eval_{enroll,test}_proc_audio \
		   data/sre21_audio-visual_eval_test_proc_audio
fi

if [ $stage -le 4 ];then
		  
  echo "Remove segments shorter than 2secs"
  hyperion-dataset remove_short_segments \
		   --dataset data/${nnet_data}_proc_audio \
		   --output-dataset data/${nnet_data}_filtered \
		   --length-name duration --min-length 2.0

  echo "Remove speakers with less than 4 audios"
  hyperion-dataset remove_classes_few_segments \
		   --dataset data/${nnet_data}_filtered \
		   --class-name speaker --min-segs 4

  echo "Removing unnecessary segment columns"
  hyperion-tables drop_columns \
		  --input-file data/${nnet_data}_filtered/segments.csv \
		  --columns speaker gender language source_type dataset duration \
		  --keep

fi

if [ $stage -le 5 ];then
  echo "Split training data into training and validation"
  hyperion-dataset split_train_val \
		   --dataset data/${nnet_data}_filtered \
		   --val-prob 0.02 \
		   --joint-classes speaker --min-train-samples 1 \
		   --seed 1123581321 \
		   --train-dataset data/${nnet_data}_train \
		   --val-dataset data/${nnet_data}_val 
fi


if [ $stage -le 6 ];then
  echo "Prepare data to train LID model"
  		  
  echo "Remove segments shorter than 3secs"
  hyperion-dataset remove_short_segments \
		   --dataset data/${nnet_data}_proc_audio \
		   --output-dataset data/${nnet_data}_lid \
		   --length-name duration --min-length 3.0

  echo "change USE to ENG"
  awk '{ sub(/",USE,/, ",ENG,"); print $0}' data/${nnet_data}_lid/segments.csv > data/${nnet_data}_lid/tmp.csv
  mv data/${nnet_data}_lid/tmp.csv data/${nnet_data}_lid/segments.csv
  
  echo "Remove languages less than 10 audios"
  hyperion-dataset remove_classes_few_segments \
		   --dataset data/${nnet_data}_lid \
		   --class-name speaker --min-segs 10


  echo "Remove segments with uncertain langs"
  hyperion-dataset remove_class_ids \
		   --dataset data/${nnet_data}_lid \
		   --class-name language \
		   --rebuild-idx \
		   --remove-na \
		   --class-ids BEN.HIN BEN.INE \
		   CMN.JPN CMN.JPN.WUU CMN.THA.WUU CMN.WUU CMN.YUE \
		   HIN.INE HIN.INE.PAN HIN.INE.PAN.URD HIN.INE.PNB \
		   HIN.INE.TAM HIN.INE.URD HIN.KHM.URD HIN.PAN \
		   HIN.PAN.URD HIN.TAM HIN.URD \
		   INE.TAM INE.URD \
		   ITA.SPA \
		   NAN.CMN NAN.TGL \
		   PAN.URD other USE

  echo "Removing unnecessary segment columns"
  hyperion-tables drop_columns \
		  --input-file data/${nnet_data}_lid/segments.csv \
		  --columns language source_type duration \
		  --keep
fi

if [ $stage -le 7 ];then
  echo "Split LID training data into training and validation"
  hyperion-dataset split_train_val \
		   --dataset data/${nnet_data}_lid \
		   --val-prob 0.02 \
		   --joint-classes language --min-train-samples 1 \
		   --seed 1123581321 \
		   --train-dataset data/${nnet_data}_lid_train \
		   --val-dataset data/${nnet_data}_lid_val 
fi

if [ $stage -le 8 ];then
  echo "Split SRE24 dev into 2 folds"
  hyperion-dataset split_folds \
		   --dataset data/sre24_audio_dev_enroll \
		   --num-folds 2 \
		   --disjoint-classes speaker \
		   --joint-classes gender \
		   --seed 1123581321 \
		   --output-path data/sre24_audio_dev_enroll/folds

  for name in sre24_audio_dev_test sre24_audio-visual_dev_test
  do
    for fold in 0 1
    do
      for side in train test
      do
	hyperion-dataset filter_by_classes_and_enrollments \
			 --dataset data/$name \
			 --class-name speaker \
			 --class-file data/sre24_audio_dev_enroll/folds/$fold/$side/speaker.csv \
			 --enrollment-name enrollment \
			 --enrollment-file data/sre24_audio_dev_enroll/folds/$fold/$side/enrollment.csv \
			 --output-dataset data/$name/folds/$fold/$side
      done
    done
    mkdir -p data/$name/folds/0+1/test
    for file_path in $(ls data/$name/folds/0/test/trials*.csv)
    do
      file_name=$(basename $file_path)
      hyperion-merge-trials --input-files data/$name/folds/{0,1}/test/$file_name \
			    --output-file data/$name/folds/0+1/test/$file_name
    done
  done
fi

if [ $stage -le 9 ];then
  echo  "temporal fix of folds trials until I fix the python code"
  for name in sre24_audio_dev_test
  do
    for fold in 0+1
    do
      for side in test
      do
	for trials in trials trials_ext
	do
	  awk -v key=data/$name/${trials}.tsv '
BEGIN{ 
  FS="\t"; OFS="\t"; 
  while(getline < key)
  {
    v[$1$2]=$0;
  };
  FS=",";
}
{ print v[$1$2] }' data/$name/folds/$fold/$side/${trials}.csv > data/$name/folds/$fold/$side/${trials}.tsv
	done
      done
    done
  done
  for name in sre24_audio-visual_dev_test
  do
    for fold in 0+1
    do
      for side in test
      do
	for trials in trials trials_ext
	do
	  awk -v key=data/$name/${trials}.tsv '
BEGIN{ 
  FS="\t"; OFS="\t"; 
  while(getline < key)
  {
    v[$1$3]=$0;
  };
  FS=",";
}
{ print v[$1$2] }' data/$name/folds/$fold/$side/${trials}.csv > data/$name/folds/$fold/$side/${trials}.tsv
	done
      done
    done
  done

fi


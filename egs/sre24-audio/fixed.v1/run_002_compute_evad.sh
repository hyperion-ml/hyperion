#!/bin/bash
# Copyright
#                2018   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e
nodes=fs06
vad_dir=`pwd`/exp/vad_e
vad_config=conf/vad_16k.yaml
nj=40

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ -z "$vad_config" ];then
  echo "We are not using VAD in this configuration"
  exit 0
fi

datasets="sre_cts_superset
sre16_eval_train
sre21_audio-visual_dev_test
sre21_audio-visual_eval_test
sre21_audio_dev_enroll
sre21_audio_dev_test
sre21_audio_eval_enroll
sre21_audio_eval_test
sre24_audio_dev_enroll
sre24_audio_dev_test
sre24_audio-visual_dev_test
sre24_audio_eval_enroll
sre24_audio_eval_test
sre24_audio-visual_eval_test"


if [ $stage -le 1 ]; then
  # Prepare to distribute data over multiple machines
  # This only does something at CLSP grid
  for name in sre_cts_superset sre16_eval_train
  do
    hyp_utils/create_data_split_dirs.sh \
      $vad_dir/$name \
      $USER/hyp-data/sre24-audio/fixed.v1/vad $nodes
  done
fi

#Train datasets
if [ $stage -le 2 ];then
  for name in $datasets
  do
    # This creates links to distribute data in CLSP grid
    # If you are not at CLSP grid, it does nothing and can be deleted
    hyp_utils/create_data_split_links.sh $vad_dir/$name/vad.JOB.ark $nj
    echo "compute vad for $name"
    $train_cmd JOB=1:$nj $vad_dir/$name/log/vad.JOB.log \
	       hyp_utils/conda_env.sh \
	       hyperion-compute-energy-vad --cfg $vad_config \
	       --recordings-file data/$name/recordings.csv \
	       --output-spec ark,csv:$vad_dir/$name/vad.JOB.ark,$vad_dir/$name/vad.JOB.csv \
	       --part-idx JOB --num-parts $nj || exit 1

    hyperion-tables cat \
		    --table-type vads \
		    --output-file $vad_dir/$name/vad.csv --num-tables $nj
    hyperion-dataset add_vads \
		     --dataset data/$name \
		     --vads-name vad \
		     --vads-file $vad_dir/$name/vad.csv
  done
fi

if [ $stage -le 3 ];then
  for name in sre24_audio_dev_enroll sre24_audio_eval_enroll
  do
    echo "Convert $name time-marks to binary vad format"
    hyperion-convert-vad-format time_marks_to_bin \
				--in-vad-file csv:data/$name/target_speaker.csv \
				--out-vad-file ark,csv:data/$name/target_speaker_binary.ark,data/$name/target_speaker_binary.csv \
				--frame-length 25. --frame-shift 10. \
				--segments-file data/$name/segments.csv
    hyperion-dataset add_vads \
		     --dataset data/$name \
		     --vads-name target_speaker_binary \
		     --vads-file data/$name/target_speaker_binary.csv

    echo "Replace $name energy vad by the target-speaker ground truth vad if it exists"
    hyperion-tables replace_columns \
		    --input-file data/$name/vad.csv \
		    --replacement-file data/$name/target_speaker_binary.csv \
		    --output-file data/$name/vad_mixed.csv

    hyperion-dataset add_vads \
		     --dataset data/$name \
		     --vads-name vad_mixed \
		     --vads-file data/$name/vad_mixed.csv
    
  done
fi

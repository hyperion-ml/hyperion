#!/bin/bash
# Copyright
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
nj=10
config_file=default_config.sh
. parse_options.sh || exit 1;
. $config_file
. datapath.sh

# We prepare the noise files and RIR for online speech augmentation
if [ $stage -le 1 ]; then
  for name in noise music
  do
    hyperion-prepare-data musan \
			  --corpus-dir $musan_root \
			  --subset $name \
			  --output-dir data/musan_$name
  done
fi

if [ $stage -le 2 ]; then
  # # Prepare to distribute data over multiple machines
  # # This only does something at CLSP grid
  # hyp_utils/create_data_split_dirs.sh $vad_dir $USER/hyp-data/voxceleb/v1.2/vad $nodes

  for name in musan_noise musan_music
  do
    input_data_dir=data/$name
    output_data_dir=data/${name}_proc_audio
    output_dir=exp/proc_audio/$name
    $train_cmd JOB=1:$nj $output_dir/log/preproc_audios_${name}.JOB.log \
	       hyp_utils/conda_env.sh \
	       hyperion-preprocess-audio-files \
	       --audio-format flac  \
	       --part-idx JOB --num-parts $nj \
	       --recordings-file $input_data_dir/recordings.csv \
	       --output-path $output_dir \
	       --output-recordings-file $output_dir/recordings.JOB.csv
    
    hyperion-tables cat \
		    --table-type recordings \
		    --output-file $output_dir/recordings.csv --num-tables $nj
    hyperion-dataset set_recordings \
		     --dataset $input_data_dir \
		     --recordings-file $output_dir/recordings.csv \
		     --output-dataset $output_data_dir
    
    
  done
fi

if [ $stage -le 3 ]; then
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi
  hyperion-prepare-data rirs --corpus-dir RIRS_NOISES/simulated_rirs/smallroom --output-dir data/rirs_smallroom
  hyperion-prepare-data rirs --corpus-dir RIRS_NOISES/simulated_rirs/mediumroom --output-dir data/rirs_mediumroom
  hyperion-prepare-data rirs --corpus-dir RIRS_NOISES/real_rirs_isotropic_noises --output-dir data/rirs_real
  for rirs in rirs_smallroom rirs_mediumroom rirs_real
  do
    output_dir=exp/rirs/$rirs
    data_dir=data/$rirs
    $train_cmd $output_dir/log/pack_rirs_${name}.log \
	       hyp_utils/conda_env.sh \
	       hyperion-pack-wav-rirs ${args} --input $data_dir/recordings.csv \
	       --output h5,csv:$output_dir/rirs.h5,$output_dir/rirs.csv || exit 1;
    hyperion-dataset add_features --dataset $data_dir \
		     --features-name rirs --features-file $output_dir/rirs.csv

  done
fi


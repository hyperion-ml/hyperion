#!/bin/bash
#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.

# Make lists for using DIHARDII data for training PLDA and x-vector

if [  $# != 3 ]; then
    echo "$0 <db-path> <dev/eval> <output_path>"
    exit 1
fi

db_path=$1
dev_eval=$2
output_path=$3


data_name=dihard2_train_${dev_eval}
output_path=$output_path/$data_name

mkdir -p $output_path

#concatenate rttm files
db_path2=$db_path/data/single_channel
rttm_path=$db_path2/rttm
uem=$db_path2/uem/all.uem
wav_path=$db_path2/flac

#concatenate all rttms
cat $rttm_path/*.rttm > $output_path/all.rttm


python local/make_dihard_train.py \
    --rttm $output_path/all.rttm \
    --wav-path $wav_path \
    --output-path $output_path \
    --data-prefix dihard2-${dev_eval}-

    #--uem $uem \
    
    
#make spk2utt so kaldi don't complain
utils/utt2spk_to_spk2utt.pl $output_path/utt2spk \
    > $output_path/spk2utt

utils/fix_data_dir.sh $output_path



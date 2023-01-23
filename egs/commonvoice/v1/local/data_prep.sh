#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <language> <src-dir> <dst-dir>"
  echo "e.g.: $0 ${language} /export/c06/ylu125/GSP/corpora/CommonVoice data/"
  exit 1
fi

language=$1
src=$2
dst=$3

if [ ! -d $src/cv-corpus-12.0-2022-12-07/${language} ]; then
  wget https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-12.0-2022-12-07/cv-corpus-12.0-2022-12-07-${language}.tar.gz 
  tar -xvzf cv-corpus-12.0-2022-12-07-${language}.tar.gz -C $src
  rm cv-corpus-12.0-2022-12-07-${language}.tar.gz 
fi


lhotse prepare commonvoice -l ${language} $src/cv-corpus-12.0-2022-12-07/ ${dst}/${language}


for part in dev test train
do
  lhotse kaldi export ${dst}/${language}/cv-${language}_recordings_${part}.jsonl.gz ${dst}/${language}/cv-${language}_supervisions_${part}.jsonl.gz  ${dst}/${language}_${part}
  utils/utt2spk_to_spk2utt.pl ${dst}/${language}_${part}/utt2spk > ${dst}/${language}_${part}/spk2utt
  utils/fix_data_dir.sh ${dst}/${language}_${part} 
  steps_xvec/audio_to_duration.sh --cmd "$train_cmd" ${dst}/${part//-/_}
done


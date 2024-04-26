# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  librispeech_root=/export/corpora5/LibriSpeech 
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  librispeech_root=/export/common/data/corpora/ASR/openslr/SLR12/LibriSpeech
  musan_root=/export/common/data/corpora/MUSAN/musan
else
  echo "Put your database paths here"
  exit 1
fi



# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  librispeech_root=/export/corpora5/LibriSpeech 
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  # voxceleb1_root=/expscratch/dsnyder/VoxCeleb1 #voxceleb1 v1
  # voxceleb1_root=/exp/jvillalba/corpora/voxceleb1 #voxceleb1 v2
  # voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
  # musan_root=/expscratch/dgromero/corpora-open/musan
  echo "Put your database paths here"
  exit 1
else
  echo "Put your database paths here"
  exit 1
fi



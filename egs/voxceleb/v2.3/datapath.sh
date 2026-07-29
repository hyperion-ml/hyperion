# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname -y)" == "clsp" ];then
  # voxceleb1_root=/export/corpora5/VoxCeleb1_v1 #voxceleb1 v1
  voxceleb1_root=/export/corpora5/VoxCeleb1_v2 #voxceleb1 v2
  voxceleb2_root=/export/corpora5/VoxCeleb2
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  # voxceleb1_root=/expscratch/dsnyder/VoxCeleb1 #voxceleb1 v1
  voxceleb1_root=/exp/jvillalba/corpora/voxceleb1 #voxceleb1 v2
  voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
  voxsrc22_root=/exp/jvillalba/corpora/voxsrc22
  musan_root=/expscratch/dgromero/corpora-open/musan
else
  echo "Put your database paths here"
  exit 1
fi



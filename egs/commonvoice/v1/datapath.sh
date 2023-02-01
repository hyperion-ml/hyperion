# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  commonvoice_root=
  musan_root=/export/corpora5/JHU/musan
  echo "Put your database paths here"
  exit 1
elif [ "$(hostname --domain)" == "rockfish.cluster" ];then
  commonvoice_root=/data/jvillal7/corpora/commonvoice
  musan_root=/data/jvillal7/corpora/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  echo "Put your database paths here"
  exit 1
else
  echo "Put your database paths here"
  exit 1
fi



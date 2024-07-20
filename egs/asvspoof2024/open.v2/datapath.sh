# Copyright
#            2024   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname -y)" == "clsp" ];then
  asvspoof_root=/export/fs06/corpora8/asvspoof
  asvspoof2015_root=$asvspoof_root/2015
  asvspoof2017_root=$asvspoof_root/2017
  asvspoof2019_root=$asvspoof_root/2019
  asvspoof2021_root=$asvspoof_root/2021
  asvspoof2024_root=$asvspoof_root/2024
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  echo "Put your database paths here"
  exit 1
  musan_root=/expscratch/dgromero/corpora-open/musan
else
  echo "Put your database paths here"
  exit 1
fi



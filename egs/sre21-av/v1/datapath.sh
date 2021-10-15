# Copyright
#            2021   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  ldc_root=/export/corpora5/LDC
  ldc_root3=/export/corpora3/LDC
  janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
  sre21_dev_root=$ldc_root/LDC2021E09
  sre21_eval_root=$ldc_root/LDC2021E10
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  ldc_root=/export/common/data/corpora/LDC
  sre_root=/export/common/data/corpora/NIST/SRE
  janus_root=$sre_root/SRE19/LDC2019E55_Janus_Multimedia_Dataset
  sre21_dev_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E09
  sre21_eval_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E10
else
  echo "Put your database paths here"
  exit 1
fi




# Copyright
#            2021   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --y)" == "clsp" ];then
  ldc_root=/export/corpora5/LDC
  ldc_root3=/export/corpora3/LDC
  ldc_root6=/export/corpora6/LDC
  ldc_root7=/export/fs05/corpora7/LDC
  voxceleb1_root=/export/corpora5/VoxCeleb1_v2
  voxceleb2_root=/export/corpora5/VoxCeleb2
  sre16_root=$ldc_root/LDC2019S20
  # sre16_dev_root=$ldc_root/LDC2019S20/data/dev/R148_0_0
  # sre16_eval_root=$ldc_root/LDC2019S20/data/eval/R149_0_1
  janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
  sre_superset_root=$ldc_root6/LDC2021E08
  sre21_dev_root=$ldc_root6/LDC2021E09
  sre21_eval_root=$ldc_root6/LDC2021E10
  sre24_dev_root=$ldc_root7/LDC2024E12/sre24_devset
  sre24_dev_docs_root=$ldc_root7/LDC2024E34/sre24_devset_docs
  sre24_eval_root=$ldc_root7/LDC2024E11/sre24_evalset
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  ldc_root=/export/common/data/corpora/LDC
  sre_root=/export/common/data/corpora/NIST/SRE
  my_root=/exp/jvillalba/corpora
  sre16_root=$ldc_root/LDC2019S20
  janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
  sre_superset_root=$ldc_root/LDC2021E08
  sre21_dev_root=$ldc_root/LDC2021E09
  sre21_eval_root=$ldc_root/LDC2021E10
  sre24_dev_root=$my_root/LDC2024E12/sre24_devset
  sre24_dev_docs_root=$my_root/LDC2024E34/sre24_devset_docs
  sre24_eval_root=$my_root/LDC2024E11/sre24_evalset
  musan_root=/export/common/data/corpora/MUSAN/musan
  # sre16_dev_root=/exp/jvillalba/corpora/LDC2019S20/data/dev/R148_0_0
  # sre16_eval_root=/exp/jvillalba/corpora/LDC2019S20/data/eval/R149_0_1
  # janus_root=$sre_root/SRE19/LDC2019E55_Janus_Multimedia_Dataset
  # sre_superset_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E08
  # sre21_dev_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E09
  # sre21_eval_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E10
else
  echo "Put your database paths here"
  exit 1
fi




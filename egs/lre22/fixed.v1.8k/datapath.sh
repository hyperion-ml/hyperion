# Copyright
#            2021   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  ldc_root3=/export/fs02/corpora3/LDC
  ldc_root5=/export/corpora5/LDC
  ldc_root=/export/corpora6/LDC
  #voxceleb1_root=/export/corpora5/VoxCeleb1_v2
  #voxceleb2_root=/export/corpora5/VoxCeleb2
  sre16_eval_root=$ldc_root5/LDC2018E30/data/eval/R149_0_1
  #janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
  #sre_superset_root=$ldc_root/LDC2021E08
  #sre21_dev_root=$ldc_root/LDC2021E09
  #sre21_eval_root=$ldc_root/LDC2021E10
  lre17_train_root=$ldc_root/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets
  lre17_eval_root=$ldc_root/LDC2022E17_2017_NIST_Language_Recognition_Evaluation_Test_Set
  lre22_dev_root=$ldc_root/LDC2022E14_2022_NIST_Language_Recognition_Evaluation_Development_Data
  lre22_eval_root=/export/corpora6/lre22_test_data_v2
  voxlingua_root=/export/corpora6/voxlingua107
  musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  ldc_root=/export/common/data/corpora/LDC
  sre_root=/export/common/data/corpora/NIST/SRE
  my_root=/exp/jvillalba/corpora
  #voxceleb1_root=/exp/jvillalba/corpora/voxceleb1
  #voxceleb2_root=/expscratch/dgromero/corpora/vox2
  sre16_dev_root=/exp/jvillalba/corpora/LDC2019S20/data/dev/R148_0_0
  sre16_eval_root=/exp/jvillalba/corpora/LDC2019S20/data/eval/R149_0_1
  #janus_root=$sre_root/SRE19/LDC2019E55_Janus_Multimedia_Dataset
  sre_superset_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E08
  sre21_dev_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E09
  sre21_eval_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E10
  lre17_train_root=$my_root/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets
  lre17_eval_root=$my_root/LDC2022E17_2017_NIST_Language_Recognition_Evaluation_Test_Set
  lre22_dev_root=$my_root/LDC2022E14_2022_NIST_Language_Recognition_Evaluation_Development_Data
  lre22_eval_root=$my_root/lre22_test_data_v2
  voxlingua_root=$my_root/voxlingua107
  musan_root=/expscratch/dgromero/corpora/musan
else
  echo "Put your database paths here"
  exit 1
fi

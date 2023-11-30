# Copyright
#            2022   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  ldc_root3=/export/fs02/corpora3/LDC
  ldc_root5=/export/corpora5/LDC
  ldc_root=/export/corpora6/LDC
  sre16_dev_root=$ldc_root/LDC2019S20/data/dev/R148_0_0
  sre16_eval_root=$ldc_root/LDC2019S20/data/eval/R149_0_1
  sre18_dev_root=$ldc_root5/LDC2018E46
  sre18_eval_root=$ldc_root3/LDC2018E51
  sre19cmn2_eval_root=$ldc_root3/LDC2019E58
  sre_superset_root=$ldc_root/LDC2021E08
  sre21_dev_root=$ldc_root/LDC2021E09
  sre21_eval_root=$ldc_root/LDC2021E10
  lre17_train_root=$ldc_root/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets
  lre17_eval_root=$ldc_root/LDC2022E17_2017_NIST_Language_Recognition_Evaluation_Test_Set
  lre22_dev_root=$ldc_root/LDC2022E14_2022_NIST_Language_Recognition_Evaluation_Development_Data
  lre22_eval_root=/export/corpora6/lre22_test_data_v2
  voxlingua_root=/export/corpora6/voxlingua107
  musan_root=/export/corpora5/JHU/musan
  babel_assamese_root=$ldc_root/LDC2016S06
  babel_bengali_root=$ldc_root/LDC2016S08
  babel_pashto_root=$ldc_root/LDC2016S09
  babel_turkish_root=$ldc_root/LDC2016S10
  babel_georgian_root=$ldc_root/LDC2016S12
  babel_vietnam_root=$ldc_root/LDC2017S01
  babel_haitian_root=$ldc_root/LDC2017S03
  babel_lao_root=$ldc_root/LDC2017S08
  babel_tamil_root=$ldc_root/LDC2017S13
  babel_zulu_root=$ldc_root/LDC2017S19
  babel_kurmanji_root=$ldc_root/LDC2017S22
  babel_tok_root=$ldc_root/LDC2018S02
  babel_kazakh_root=$ldc_root/LDC2018S13
  babel_telugu_root=$ldc_root/LDC2018S16
  babel_lithuanian_root=$ldc_root/LDC2019S03
  fleurs_root=/export/corpora6/LRE/FLEURS2022
  lwazi_root=/export/corpora6/LRE/Lwazi2009
  nchlt_root=/export/corpora6/LRE/NCHLT2014
  ammi_root=/export/corpora6/LRE/AMMI2020
  cv20_root=/export/corpora5/mozilla-common-voice/cv-corpus-5.1-2020-06-22
  cv22_root=/export/corpora6/LRE/CommonVoice2020/cv-corpus-11.0-2022-09-21
  adi_root=/export/corpora6/ADI17
  ast_root=/export/corpora6/LRE/AST2004
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  ldc_root=/export/common/data/corpora/LDC
  sre_root=/export/common/data/corpora/NIST/SRE
  my_root=/exp/jvillalba/corpora
  sre16_dev_root=/exp/jvillalba/corpora/LDC2019S20/data/dev/R148_0_0
  sre16_eval_root=/exp/jvillalba/corpora/LDC2019S20/data/eval/R149_0_1
  sre18_dev_root=$sre_root/SRE18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
  sre18_eval_root=$sre_root/SRE18/Eval/LDC2018E51
  sre19cmn2_eval_root=/exp/jvillalba/corpora/LDC2019E58
  sre_superset_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E08
  sre21_dev_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E09
  sre21_eval_root=/exp/jvillalba/corpora/sre21/releases/LDC2021E10
  lre17_train_root=$my_root/LDC2022E16_2017_NIST_Language_Recognition_Evaluation_Training_and_Development_Sets
  lre17_eval_root=$my_root/LDC2022E17_2017_NIST_Language_Recognition_Evaluation_Test_Set
  lre22_dev_root=$my_root/LDC2022E14_2022_NIST_Language_Recognition_Evaluation_Development_Data
  lre22_eval_root=$my_root/lre22_test_data_v2
  voxlingua_root=$my_root/voxlingua107
  musan_root=/export/common/data/corpora/MUSAN/musan
  babel_assamese_root=$ldc_root/LDC2016S06
  babel_bengali_root=$ldc_root/LDC2016S08
  babel_pashto_root=$ldc_root/LDC2016S09
  babel_turkish_root=$my_root/LDC2016S10
  babel_georgian_root=$my_root/LDC2016S12
  babel_vietnam_root=$my_root/LDC2017S01
  babel_haitian_root=$my_root/LDC2017S03
  babel_lao_root=$ldc_root/LDC2017S08
  babel_tamil_root=$ldc_root/LDC2017S13
  babel_zulu_root=$ldc_root/LDC2017S19
  babel_kurmanji_root=$ldc_root/LDC2017S22
  babel_tok_root=$my_root/LDC2018S02
  babel_kazakh_root=$ldc_root/LDC2018S13
  babel_telugu_root=$ldc_root/LDC2018S16
  babel_lithuanian_root=$my_root/LDC2019S03
  adi_root=/exp/jvillalba/corpora/ADI17

else
  echo "Put your database paths here"
  exit 1
fi

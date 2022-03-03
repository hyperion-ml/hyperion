# Copyright
#            2019   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases
if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  ldc_root=/export/corpora5/LDC
  sre19_dev_root=$ldc_root/LDC2019E56
  sre19_eval_root=$ldc_root/LDC2019E57
  janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
elif [ "$(hostname --domain)" == "cm.gemini" ];then
  ldc_root=/export/common/data/corpora/LDC
  sre_root=/export/common/data/corpora/NIST/SRE
  sre19_dev_root=$sre_root/SRE19/LDC2019E56
  sre19_eval_root=$sre_root/SRE19/LDC2019E57
  janus_root=$sre_root/SRE19/LDC2019E55_Janus_Multimedia_Dataset
else
  echo "Put your database paths here"
  exit 1
fi

#trial files

# SRE19 trials
sre19_dev_trials_av=data/sre19av_v_dev_test/trials
sre19_eval_trials_av=data/sre19av_v_eval_test/trials

# Janus trials
janus_trials=data/janus_test/trials



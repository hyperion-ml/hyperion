# Paths to the databases used in the experiment
#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
  
  adi_root=/export/corpora6/ADI17
  musan_root=/export/corpora5/JHU/musan

  

elif [ "$(hostname --domain)" == "kam.local" ];then

  adi_root=/Users/kamstudy/Documents/Corpora/ADI17
  exit 1

  fi

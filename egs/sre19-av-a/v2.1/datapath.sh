# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#master key
master_key_dir=master_key_sre04-12
master_key=$master_key_dir/NIST_SRE_segments_key.v2.csv

#paths to databases

if [ "$(hostname -y)" == "clsp" ];then
    ldc_root=/export/corpora5/LDC
    ldc_root3=/export/corpora3/LDC
    sitw_root=/export/corpora5/SRI/SITW
    sre08sup_root=$ldc_root/LDC2011S11
    sre10_root=/export/corpora5/SRE/SRE2010/eval
    sre10_root=$ldc_root3/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root3/LDC2012E09/SRE10_16K
    sre12_root=$ldc_root3/LDC2016E45
    voxceleb1_root=/export/corpora5/VoxCeleb1_v1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    sre18_dev_root=$ldc_root/LDC2018E46
    sre18_eval_root=$ldc_root3/LDC2018E51
    sre19_dev_root=$ldc_root/LDC2019E56
    sre19_eval_root=$ldc_root/LDC2019E57
    janus_root=$ldc_root/LDC2019E55/Janus_Multimedia_Dataset
    musan_root=/export/corpora5/JHU/musan
    dihard2_dev=$ldc_root/LDC2019E31/LDC2019E31_Second_DIHARD_Challenge_Development_Data
    dihard2_eval=$ldc_root/LDC2019E32/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    ldc_root=/export/common/data/corpora/LDC
    sre_root=/export/common/data/corpora/NIST/SRE
    sitw_root=$sre_root/sitw_database.v4
    sre08sup_root=/exp/jvillalba/corpora/LDC2011S11
    sre10_root=$ldc_root/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root/LDC2012E09/SRE10_16K
    sre12_root=$sre_root/SRE2012
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
    sre18_dev_root=$sre_root/SRE18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
    sre18_eval_root=$sre_root/SRE18/Eval/LDC2018E51
    sre19_dev_root=$sre_root/SRE19/LDC2019E56
    sre19_eval_root=$sre_root/SRE19/LDC2019E57
    janus_root=$sre_root/SRE19/LDC2019E55_Janus_Multimedia_Dataset
    musan_root=/expscratch/dgromero/corpora-open/musan
    dihard2_dev=/export/common/data/corpora/LDC/LDC2019E31
    dihard2_eval=/export/common/data/corpora/LDC/LDC2019E32/v1.1/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
else
    echo "Put your database paths here"
    exit 1
fi


#trial files
# SITW Trials
sitw_dev_trials=data/sitw_dev_test/trials
sitw_eval_trials=data/sitw_eval_test/trials
sitw_conds=(core-core core-multi assist-core assist-multi)

# SRE18 trials
sre18_dev_trials_vast=data/sre18_dev_test_vast/trials
sre18_eval_trials_vast=data/sre18_eval_test_vast/trials

# SRE19 trials
sre19_dev_trials_av=data/sre19av_a_dev_test/trials
sre19_eval_trials_av=data/sre19av_a_eval_test/trials

# Janus trials
janus_trials=data/janus_test/trials



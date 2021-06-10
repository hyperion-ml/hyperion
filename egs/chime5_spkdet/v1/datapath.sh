# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    ldc_root=/export/corpora5/LDC
    ldc_root3=/export/corpora3/LDC
    sitw_root=/export/corpora5/SRI/sitw
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
    chime5_root=/export/corpora4/dgr_CHiME5_segmented
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
    chime5_root=/expscratch/dgromero/corpora/chime5/dgr_CHiME5_segmented
    mx6_root=$ldc_root/LDC2013S03
else
    echo "Put your database paths here"
    exit 1
fi




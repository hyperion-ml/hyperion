# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#master key
master_key_dir=master_key_sre04-12
master_key=$master_key_dir/NIST_SRE_segments_key.v2.csv

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    ldc_root=/export/corpora5/LDC
    ldc_root3=/export/corpora3/LDC
    sitw_root=/export/corpora5/SRI/sitw
    swbd_cell2_root=$ldc_root/LDC2004S07
    swbd2_ph1_root=$ldc_root3/LDC98S75
    sre08sup_root=$ldc_root/LDC2011S11
    sre10_root=/export/corpora5/SRE/SRE2010/eval
    sre10_root=$ldc_root3/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root3/LDC2012E09/SRE10_16K
    sre12_root=$ldc_root3/LDC2016E45
    voxceleb1_root=/export/corpora5/VoxCeleb1_v1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    sre16_dev_root=$ldc_root3/LDC2016E46
    sre16_eval_root=$ldc_root/LDC2018E30/data/eval/R149_0_1
    sre18_dev_root=$ldc_root/LDC2018E46
    sre18_eval_root=$ldc_root3/LDC2018E51
    sre18_dev_meta=${sre18_dev_root}/docs/sre18_dev_segment_key.tsv
    sre19cmn2_eval_root=$ldc_root3/LDC2019E58
    musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    ldc_root=/export/common/data/corpora/LDC
    sre_root=/export/common/data/corpora/NIST/SRE
    swbd_cell2_root=$ldc_root/LDC2004S07
    swbd2_ph1_root=$ldc_root/LDC98S75
    sitw_root=$sre_root/sitw_database.v4
    sre08sup_root=/exp/jvillalba/corpora/LDC2011S11
    sre10_root=$ldc_root/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root/LDC2012E09/SRE10_16K
    sre12_root=$sre_root/SRE2012
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora/vox2
    sre16_dev_root=$sre_root/LDC2016E46_SRE16_Call_My_Net_Training_Data/
    sre16_eval_root=$sre_root/SRE16_eval
    sre18_dev_root=$sre_root/SRE18/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set
    sre18_eval_root=$sre_root/SRE18/Eval/LDC2018E51
    sre18_dev_meta=${sre18_dev_root}/docs/sre18_dev_segment_key.tsv
    sre19cmn2_eval_root=/exp/jvillalba/corpora/LDC2019E58
    musan_root=/expscratch/dgromero/corpora/musan
else
    echo "Put your database paths here"
    exit 1
fi


# SRE18 trials
sre18_dev_trials_cmn2=data/sre18_dev_test_cmn2/trials
sre18_eval_trials_cmn2=data/sre18_eval_test_cmn2/trials

# SRE19 trials
sre19_eval_trials_cmn2=data/sre19_eval_test_cmn2/trials




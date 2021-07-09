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
    swbd_cell2_root=$ldc_root/LDC2004S07
    swbd2_ph1_root=$ldc_root3/LDC98S75
    sre08sup_root=$ldc_root/LDC2011S11
    sre10_root=/export/corpora5/SRE/SRE2010/eval
    sre10_root=$ldc_root3/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root3/LDC2012E09/SRE10_16K
    sre12_root=$ldc_root3/LDC2016E45
    voxceleb1_root=/export/corpora5/VoxCeleb1_v1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    musan_root=/export/corpora5/JHU/musan
    callhome_root=$ldc_root/LDC2001S97
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    ldc_root=/export/common/data/corpora/LDC
    sre_root=/export/common/data/corpora/NIST/SRE
    swbd_cell2_root=$ldc_root/LDC2004S07
    swbd2_ph1_root=$ldc_root/LDC98S75
    sre08sup_root=/exp/jvillalba/corpora/LDC2011S11
    sre10_root=$ldc_root/LDC2012E09/SRE10/eval
    sre10_16k_root=$ldc_root/LDC2012E09/SRE10_16K
    sre12_root=$sre_root/SRE2012
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora/vox2
    musan_root=/expscratch/dgromero/corpora/musan
    callhome_root=$ldc_root/LDC2001S97
else
    echo "Put your database paths here"
    exit 1
fi





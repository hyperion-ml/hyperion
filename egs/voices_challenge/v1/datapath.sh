# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment


if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    ldc_root=/export/corpora5/LDC
    sitw_root=/export/corpora5/SRI/sitw
    voxceleb1_root=/export/corpora5/VoxCeleb1_v1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    musan_root=/export/corpora5/JHU/musan
    voices_root=/export/corpora5/SRI/VOiCES_challenge
    mx6_root=$ldc_root/LDC2013S03
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    ldc_root=/export/common/data/corpora/LDC
    sre_root=/export/common/data/corpora/NIST/SRE
    sitw_root=$sre_root/sitw_database.v4
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
    musan_root=/expscratch/dgromero/corpora-open/musan
    voices_root=/exp/jvillalba/corpora/VOiCES_challenge
    mx6_root=$ldc_root/LDC2013S03
else
    echo "Put your database paths here"
    exit 1
fi



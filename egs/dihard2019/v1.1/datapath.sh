# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#master key
master_key_dir=master_key_sre04-12
master_key=$master_key_dir/NIST_SRE_segments_key.v2.csv

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    voxceleb1_root=/export/corpora5/VoxCeleb1_v1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    musan_root=/export/corpora5/JHU/musan
    dihard2019_dev=/export/corpora5/LDC/LDC2019E31/LDC2019E31_Second_DIHARD_Challenge_Development_Data
    dihard2019_eval=/export/corpora5/LDC/LDC2019E32/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
    musan_root=/expscratch/dgromero/corpora-open/musan
    dihard2019_dev=/export/common/data/corpora/LDC/LDC2019E31
    dihard2019_eval=/export/common/data/corpora/LDC/LDC2019E32/v1.1/LDC2019E32_Second_DIHARD_Challenge_Evaluation_Data_V1.1
else
    echo "Put your database paths here"
    exit 1
fi



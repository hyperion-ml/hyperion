# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#master key
master_key_dir=master_key_sre04-12
master_key=$master_key_dir/NIST_SRE_segments_key.v2.csv

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    sdsv_root=/export/corpora5/sdsv_challenge
    voxceleb1_root=/export/corpora/VoxCeleb1
    voxceleb2_root=/export/corpora/VoxCeleb2
    musan_root=/export/corpora/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
    musan_root=/expscratch/dgromero/corpora-open/musan
else
    echo "Put your database paths here"
    exit 1
fi



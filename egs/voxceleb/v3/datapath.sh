# Copyright
#            2018   Johns Hopkins University (Author: Jesus Villalba)
#
# Paths to the databases used in the experiment

#paths to databases

if [ "$(hostname --domain)" == "clsp.jhu.edu" ];then
    voxceleb1_root=/export/corpora5/VoxCeleb1
    voxceleb2_root=/export/corpora5/VoxCeleb2
    musan_root=/export/corpora5/JHU/musan
elif [ "$(hostname --domain)" == "cm.gemini" ];then
    voxceleb1_root=/expscratch/dsnyder/VoxCeleb1
    voxceleb2_root=/expscratch/dgromero/corpora-open/vox2
    musan_root=/expscratch/dgromero/corpora-open/musan
else
    echo "Put your database paths here"
    exit 1
fi



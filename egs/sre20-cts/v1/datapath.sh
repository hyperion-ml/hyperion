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
    cnceleb_root=/export/corpora5/CN-Celeb
    sre16_dev_root=$ldc_root3/LDC2016E46
    sre16_eval_root=$ldc_root/LDC2018E30/data/eval/R149_0_1
    sre18_dev_root=$ldc_root/LDC2018E46
    sre18_eval_root=$ldc_root3/LDC2018E51
    sre18_dev_meta=${sre18_dev_root}/docs/sre18_dev_segment_key.tsv
    sre19cmn2_eval_root=$ldc_root3/LDC2019E58
    sre20cts_eval_root=$ldc_root/LDC2020E28
    cv_root=/export/corpora5/mozilla-common-voice/cv-corpus-5.1-2020-06-22
    musan_root=/export/corpora5/JHU/musan
    babel_root0=/export/corpora5/babel_CLSP
    declare -A babel_root
    babel_root["cantonese"]=$babel_root0/101-cantonese/release-current
    babel_root["assamese"]=$babel_root0/102-assamese/release-current
    babel_root["bengali"]=$babel_root0/103-bengali/release-current
    babel_root["pashto"]=$babel_root0/104-pashto/release-babel104b-v04.aY
    babel_root["turkish"]=$babel_root0/105-turkish/release-current
    babel_root["tagalog"]=$babel_root0/106-tagalog/release-current
    babel_root["vietnamese"]=$babel_root0/107-vietnamese/release-current
    babel_root["haitian"]=$babel_root0/201-haitian/release-current
    babel_root["swahili"]=$babel_root0/202-swahili/IARPA-babel202b-v1.0d-build/BABEL_OP2_202
    babel_root["lao"]=$babel_root0/203-lao/release-current
    babel_root["tamil"]=$babel_root0/204-tamil/release-current
    babel_root["kurmanji"]=$babel_root0/205-kurmanji/IARPA-babel205b-v1.0a-build/BABEL_OP2_205
    babel_root["zulu"]=$babel_root0/206-zulu/release-current
    babel_root["tokpisin"]=$babel_root0/207-tokpisin/IARPA-babel207b-v1.0e-build/BABEL_OP2_207
    babel_root["cebuano"]=$babel_root0/301-cebuano/IARPA-babel301b-v2.0b-build/BABEL_OP2_301
    babel_root["kazakh"]=$babel_root0/302-kazakh/IARPA-babel302b-v1.0a-build/BABEL_OP2_302
    babel_root["telugu"]=$babel_root0/303-telugu/IARPA-babel303b-v1.0a-build/BABEL_OP2_303
    babel_root["lithuanian"]=$babel_root0/304-lithuanian/IARPA-babel304b-v1.0b-build/BABEL_OP2_304
    babel_root["guarani"]=$babel_root0/305-guarani/IARPA-babel305b-v1.0b-build/BABEL_OP3_305
    babel_root["igbo"]=$babel_root0/306-igbo/IARPA-babel306b-v2.0c-build/BABEL_OP3_306
    babel_root["amharic"]=$babel_root0/307-amharic/IARPA-babel307b-v1.0b-build/BABEL_OP3_307
    babel_root["mongolian"]=$babel_root0/401-mongolian/IARPA-babel401b-v2.0b-build/BABEL_OP3_401
    babel_root["javanese"]=$babel_root0/402-javanese/IARPA-babel402b-v1.0b-build/BABEL_OP3_402
    babel_root["dholuo"]=$babel_root0/403-dholuo/IARPA-babel403b-v1.0b-build/BABEL_OP3_403
    babel_root["georgian"]=$babel_root0/404-georgian/release-current/BABEL_OP3_404
    fisher_spa_root=$ldc_root/LDC2010S01
    lre17_train_root=$ldc_root/LDC2017E22_2017_NIST_Language_Recognition_Evaluation_Training_Data
    lre17_dev_root=$ldc_root/LDC2017E23_2017_NIST_Language_Recognition_Evaluation_Development_Data
    lre17_eval_root=$ldc_root/LDC2017E23_2017_NIST_Language_Recognition_Evaluation_Eval_Data
    mls_root=/export/corpora5/MLS
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


cv_langs="en de fr rw ca es kab fa it eu pl ru eo zh-TW zh-CN zh-HK pt nl cs \
    		   tt et fy-NL uk tr ta ky ar mn br id mt el dv sv-SE ja rm-sursilv \
    		   sl cv ro lv ia sah or cnh ga-IE ka rm-vallader as vi"
cv_datasets="$(echo $cv_langs | awk '{for(i=1;i<=NF;i++){ $i=sprintf("cvcat_%s_tel",$i)}; print $0}')"
cv_noeng_datasets="$(echo $cv_langs | awk '{for(i=2;i<=NF;i++){ $(i-1)=sprintf("cvcat_%s_tel",$i)}; $NF=""; print $0}')"

babel_langs="${!babel_root[@]}"
babel_datasets="$(echo $babel_langs | awk '{for(i=1;i<=NF;i++){ $i=sprintf("babel_%s",$i)}; print $0}')"

lre17_langs="ara-acm ara-apc ara-ary ara-arz \
		    por-brz qsl-pol qsl-rus \
		    spa-car spa-eur spa-lac \
		    zho-cmn zho-nan"
lre17_datasets="$(echo $lre17_langs | awk '{for(i=1;i<=NF;i++){ $i=sprintf("lre17_%s",$i)}; print $0}')"

mls_langs="dutch french german italian polish portuguese spanish"
mls_datasets="$(echo $mls_langs | awk '{for(i=1;i<=NF;i++){ $i=sprintf("mls_%s_tel",$i)}; print $0}')"


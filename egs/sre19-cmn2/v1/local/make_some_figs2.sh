#!/bin/bash

. path.sh


be=(lda150_splday125_adapt_v1_a1_mu1B0.5W0.75_a2_M500_mu1B0.75W0.5_sre_tel \
    lda150_splday125_adapt_v2_a1_mu1B0.5W0.75_a2_M600_mu0B0.5W0.5_sre_tel_sre18_cmn2_adapt_lab \
    lda200_splday150_adapt_v3_coral_mu1T0.5_a1_mu1B0.25W0.5_a2_M500_mu0B0W0_sre_tel_sre18_cmn2_adapt_lab)

ep=(10 22 33)
dir0=exp/scores/resnet34_zir_e256_arc0.3_do0_adam_lr0.01_b512_amp.v2
dir1=ft_1000_6000_sgdcos_lr0.05_b128_amp.v2
dirs2=(ft_eaffine_rege_w0.001_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w0.01_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w_1000_6000_sgdcos_lr0.01_b128_amp.v2_ep10 \
       ft_eaffine_rege_w10_1000_6000_sgdcos_lr0.01_b128_amp.v2)

dirs3=(ft_eaffine_rege_w0.001_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc0.001_we0.001_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w0.01_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc0.01_we0.01_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc0.1_we0.1_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc1_we_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       ft_eaffine_rege_w10_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc10_we10_1000_6000_sgdcos_lr0.01_b128_amp.v2)

w=(0.001 0.01 0.1 1 10)
be20=plda_snorm_cal_v1eval40
be2=plda_snorm300_cal_v1eval40
be3=plda_cal_v1eval40

dir=exp/figs/figs_ft2
mkdir -p $dir
table_file=$dir/table1
for i in 0 1 2
do
    ii=$((i+1))
    for j in 0 1 2 3 4
    do
	wj=${w[$j]}
	table_file1=${table_file}_w${wj}_be$ii-snorm.csv
	table_file2=${table_file}_w${wj}_be$ii.csv
	d=$dir0/${be[$i]}/$be2
	d2=$dir0/${be[$i]}/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh --print-header true "out-model" $d > $table_file1
	d=$dir0/${be[$i]}/$be3
	local/make_table_line_sre19cmn2.sh --print-header true "out-model" $d > $table_file2

	d=$dir0.$dir1/${be[$i]}/$be2
	d2=$dir0.$dir1/${be[$i]}/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh "+ft-out-full-length" $d >> $table_file1
	d=$dir0.$dir1/${be[$i]}/$be3
	local/make_table_line_sre19cmn2.sh "+ft-out-full-length" $d >> $table_file2

	
	d=$dir0.$dir1.${dirs2[$j]}/${be[$i]}/$be2
	d2=$dir0.$dir1.${dirs2[$j]}/${be[$i]}/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh "+ft-in-last-layer" $d >> $table_file1
	d=$dir0.$dir1.${dirs2[$j]}/${be[$i]}/$be3
	local/make_table_line_sre19cmn2.sh "+ft-in-last-layer" $d >> $table_file2
	
	for e in 0 2
	do
	    ee=${ep[$e]}
	    d=$dir0.$dir1.${dirs3[$j]}_ep$ee/${be[$i]}/$be2
	    d2=$dir0.$dir1.${dirs3[$j]}_ep$ee/${be[$i]}/$be20
	    if [ ! -d $d ];then
		d=$d2
	    fi

	    local/make_table_line_sre19cmn2.sh "+ft-in-full-nnet-ep${ee}" $d >> $table_file1
	    d=$dir0.$dir1.${dirs3[$j]}_ep$ee/${be[$i]}/$be3
	    local/make_table_line_sre19cmn2.sh "+ft-in-full-nnet-ep${ee}" $d >> $table_file2

	done
	awk -F "," 'BEGIN{
           getline; getline; 
           print "system,sre18-eer,sre18-min-dcf,sre18-act-dcf,sre19p-eer,sre19p-min-dcf,sre19p-act-dcf,sre19e-eer,sre19e-min-dcf,sre19e-act-dcf"
         } 
         { print $0}' $table_file1 > kk
	mv kk $table_file1
	awk -F "," 'BEGIN{
           getline; getline; 
           print "system,sre18-eer,sre18-min-dcf,sre18-act-dcf,sre19p-eer,sre19p-min-dcf,sre19p-act-dcf,sre19e-eer,sre19e-min-dcf,sre19e-act-dcf"
         } 
         { print $0}' $table_file2 > kk
	mv kk $table_file2

    done

done

local/make_some_figs2.py

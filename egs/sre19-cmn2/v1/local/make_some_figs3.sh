#!/bin/bash

. path.sh


be=(lda150_splday125_adapt_v1_a1_mu1B0.5W0.75_a2_M500_mu1B0.75W0.5_sre_tel \
    lda150_splday125_adapt_v2_a1_mu1B0.5W0.75_a2_M600_mu0B0.5W0.5_sre_tel_sre18_cmn2_adapt_lab \
    lda200_splday150_adapt_v3_coral_mu1T0.5_a1_mu1B0.25W0.5_a2_M500_mu0B0W0_sre_tel_sre18_cmn2_adapt_lab)

dir0=exp/scores
dirs1=(resnet34_zir_e256_arc0.3_do0_adam_lr0.01_b512_amp.v2 seresnet34_ser8_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1 tseresnet34_ser16_e256_arcs30m0.3_do0_adam_lr0.01_b512_amp.v1)
dir2=ft_1000_6000_sgdcos_lr0.05_b128_amp.v2
dirs3=(ft_eaffine_rege_w_1000_6000_sgdcos_lr0.01_b128_amp.v2_ep10 \
	   ft_eaffine_rege_w1_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
       	   ft_eaffine_rege_w1_1000_6000_sgdcos_lr0.01_b128_amp.v2)
       

dirs4=(ft_eaffine_rege_w_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc1_we_1000_6000_sgdcos_lr0.01_b128_amp.v2_ep10 \
	   ft_eaffine_rege_w1_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc1_we1_1000_6000_sgdcos_lr0.01_b128_amp.v2 \
	   ft_eaffine_rege_w1_1000_6000_sgdcos_lr0.01_b128_amp.v2.ft_reg_wenc1_we1_1000_4000_sgdcos_lr0.01_b128_amp.v2)

be20=plda_snorm_cal_v1eval40
be2=plda_snorm300_cal_v1eval40
be3=plda_cal_v1eval40

dir=exp/figs/figs_ft3
mkdir -p $dir
table_file=$dir/table1
for i in 0 1 2
do
    ii=$((i+1))
    for j in 0 1 2
    do
	table_file1=${table_file}_nnet${j}_be$ii-snorm.csv
	table_file2=${table_file}_nnet${j}_be$ii.csv
	dd=$dir0/${dirs1[$j]}/${be[$i]}
	d=$dd/$be2
	d2=$dd/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh --print-header true "out-model" $d > $table_file1
	d=$dd/$be3
	local/make_table_line_sre19cmn2.sh --print-header true "out-model" $d > $table_file2

	dd=$dir0/${dirs1[$j]}.$dir2/${be[$i]}
	d=$dd/$be2
	d2=$dd/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh "+ft-out-full-length" $d >> $table_file1
	d=$dd/$be3
	local/make_table_line_sre19cmn2.sh "+ft-out-full-length" $d >> $table_file2

	dd=$dir0/${dirs1[$j]}.$dir2.${dirs3[$j]}/${be[$i]}
	d=$dd/$be2
	d2=$dd/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi
	local/make_table_line_sre19cmn2.sh "+ft-in-last-layer" $d >> $table_file1
	d=$dd/$be3
	local/make_table_line_sre19cmn2.sh "+ft-in-last-layer" $d >> $table_file2

	dd=$dir0/${dirs1[$j]}.$dir2.${dirs4[$j]}/${be[$i]}
	d=$dd/$be2
	d2=$dd/$be20
	if [ ! -d $d ];then
	    d=$d2
	fi

	local/make_table_line_sre19cmn2.sh "+ft-in-full-nnet-ep10" $d >> $table_file1
	d=$dd/$be3
	local/make_table_line_sre19cmn2.sh "+ft-in-full-nnet-ep10" $d >> $table_file2

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

local/make_some_figs3.py

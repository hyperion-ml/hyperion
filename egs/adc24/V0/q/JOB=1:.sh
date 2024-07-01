#!/bin/bash
cd /home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
exp/scores/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1/pca_r/cosine/log/eval_dialect.JOB.log hyp_utils/conda_env.sh steps_be/score_voxceleb1.sh scp:exp/xvectors/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1/adi17/test/xvector.scp exp/be/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1//preproc.h5 data/adi17/test_proc_audio_no_sil/utt2lang score_cosine_dir --part-idx JOB --num-parts 
EOF
) >JOB=1:
time1=`date +"%s"`
 ( exp/scores/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1/pca_r/cosine/log/eval_dialect.JOB.log hyp_utils/conda_env.sh steps_be/score_voxceleb1.sh scp:exp/xvectors/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1/adi17/test/xvector.scp exp/be/fbank80_stmn_ecapatdnn512x3.adc24.4gpu.s1//preproc.h5 data/adi17/test_proc_audio_no_sil/utt2lang score_cosine_dir --part-idx JOB --num-parts  ) 2>>JOB=1: >>JOB=1:
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>JOB=1:
echo '#' Finished at `date` with status $ret >>JOB=1:
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.784692
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/JOB=1: -l hostname="[bc][01][234589]*" -V -l mem_free=4G,ram_free=4G    /home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/./q/JOB=1:.sh >>./q/JOB=1: 2>&1

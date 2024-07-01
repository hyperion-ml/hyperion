
if [ "$(hostname -d)" == "cm.gemini" ];then
    #export train_cmd="queue.pl --config conf/coe_gpu_short.conf --mem 4G"
    export train_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
    export cuda_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 20G"
    export cuda_cmd="queue.pl --config conf/coe_gpu_v100.conf --mem 40G"
    #export cuda_cmd="queue.pl --config conf/coe_gpu_rtx.conf --mem 40G"
    export cuda_eval_cmd="queue.pl --config conf/coe_gpu_short.conf --mem 4G"
    # export cuda_eval_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
else
    export train_cmd="queue.pl --mem 4G -l hostname=\"[bc][01][234589]*\" -V"
    #export cuda_cmd="sbatch --mem=40G --partition=gpu-a100 --time=520:00:00 --mail-user=mohammed-akram.khelfi.1@ens.etsmtl.ca --mail-type=ALL"
    export cuda_cmd="queue.pl --mem 40G -l hostname=\"c[01]*\" -l h_rt=520:00:00 -V -M mohammed-akram.khelfi.1@ens.etsmtl.ca "
    export cuda_eval_cmd="$train_cmd"
    # export train_cmd="sbatch --mem=4G --mail-user=mohammed-akram.khelfi.1@ens.etsmtl.ca --mail-type=ALL"
    
    # export cuda_eval_cmd="$train_cmd"
fi

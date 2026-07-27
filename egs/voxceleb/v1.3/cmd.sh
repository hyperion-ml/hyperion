# Select the submission backend used by this recipe.  The Slurm YAML file
# contains site-specific policy such as partitions, GPU selection, and
# scheduler options.  The recipe requests portable resources such as memory,
# CPU threads, and GPU count through hyperion-submit.
#
# Outside the Slurm cluster, the recipe runs synchronously with the local
# backend.  Local execution does not enforce resource requests; run stages one
# at a time when the machine cannot accommodate concurrent work.

if [ "$(hostname -d)" == "grid.cluster" ];then
  export train_cmd="hyperion-submit slurm --cfg conf/submit_coe_v100.yaml --mem 8G"
  export cuda_cmd="hyperion-submit slurm --cfg conf/submit_coe_v100.yaml --mem 32G --num-threads 9"
  export cuda_eval_cmd="$train_cmd"
else
  export train_cmd="hyperion-submit local"
  export cuda_cmd="hyperion-submit local"
  export cuda_eval_cmd="$train_cmd"
fi




# Select the submission backend used by this recipe.  The Slurm YAML file
# contains site-specific policy such as partitions, GPU selection, exclusions,
# and scheduler options.  The recipe requests portable resources such as
# memory, CPU threads, and GPU count through hyperion-submit.
#
# The three command variables retain the recipe's resource defaults:
#   train_cmd:      CPU jobs, 4 GB RAM
#   cuda_cmd:       GPU training jobs, 30 GB RAM
#   cuda_eval_cmd:  GPU evaluation jobs, 4 GB RAM
#
# Outside the Slurm cluster, all commands use hyperion-submit local and run
# synchronously in the current environment.  Run stages one at a time when
# the local machine cannot accommodate concurrent work.

# This preserves the distributed-training diagnostics used by the original
# recipe's environment setup.
export TORCH_DISTRIBUTED_DEBUG=DETAIL

if [ "$(hostname -d)" == "grid.cluster" ]; then
    submit_cfg=conf/submit_coe_v100.yaml
    export train_cmd="hyperion-submit slurm --cfg $submit_cfg --mem 4G"
    export cuda_cmd="hyperion-submit slurm --cfg $submit_cfg --mem 30G"
    export cuda_eval_cmd="hyperion-submit slurm --cfg $submit_cfg --mem 4G"
else
    export train_cmd="hyperion-submit local"
    export cuda_cmd="hyperion-submit local"
    export cuda_eval_cmd="hyperion-submit local"
fi

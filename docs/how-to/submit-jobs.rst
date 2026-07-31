Submitting recipe jobs
======================

``hyperion-submit`` is the scheduler-neutral command launcher for Hyperion
recipes.  This document defines its public contract.  The initial
implementation supports running a job on the current machine and submitting a
job to Slurm; SGE is deliberately not part of this interface.

Command line
------------

The command has a required execution backend, followed by submitter options
and a command separated with ``--``:

.. code-block:: bash

   hyperion-submit <local|slurm> --cfg submit.yaml [submit-options] -- command [command-arguments...]

The ``--`` separator is required.  Every token after it belongs to the
launched command, even when it begins with ``--``.  This prevents submitter
arguments such as ``--time`` from being confused with Hyperion program
arguments.

The initial portable submitter options are:

* ``--cfg PATH``: YAML configuration file, loaded with
  ``jsonargparse.ActionConfigFile``.
* ``--output-file PATH``: combined stdout/stderr log.  It is required.
* ``--num-gpus N``: number of GPUs required by each task.  The default is the
  configured value, normally zero.
* ``--num-threads N``: CPU threads required by each task.  The default is the
  configured value, normally one.
* ``--mem SIZE``: total memory required per allocated node, such as ``8G``.
* ``--mem-per-cpu SIZE``: memory required per allocated CPU, such as ``2G``.
* ``--time DURATION``: scheduler wall-clock limit, such as ``168:00:00``.
* ``--array NAME=START:END``: submit or run one task for every inclusive
  integer in the range.
* ``--max-jobs-run N``: maximum concurrently running tasks in a Slurm array;
  it requires ``--array``.  Local arrays always run sequentially.

``local`` and ``slurm`` accept the same portable options.  Backend-specific
options are intentionally not part of the first public interface; site policy
belongs in the YAML file. ``--mem`` and ``--mem-per-cpu`` are mutually
exclusive.  The launcher rejects an invocation that provides both, rather than
silently choosing one memory scope.

Examples
~~~~~~~~

Run one GPU training command through Slurm:

.. code-block:: bash

   hyperion-submit slurm --cfg conf/submit_coe_v100.yaml \
     --num-gpus 4 --num-threads 9 --mem 32G \
     --output-file exp/xvector/log/train.log -- \
     hyperion-train-wav2xvector resnet --cfg conf/train.yaml --num-gpus 4

Run the same command synchronously on the current machine:

.. code-block:: bash

   hyperion-submit local --cfg conf/submit_coe_v100.yaml \
     --num-gpus 4 --output-file exp/xvector/log/train.log -- \
     hyperion-train-wav2xvector resnet --cfg conf/train.yaml --num-gpus 4

Array jobs and task substitution
--------------------------------

``--array JOB=1:100`` creates 100 tasks.  ``NAME`` must be a shell-compatible
environment-variable name and both range endpoints must be positive integers
with ``START <= END``.

For every task, the launcher sets ``NAME`` in the child environment to that
task's integer value.  Before launch, it also replaces every literal occurrence
of ``NAME`` in every command argument and in ``--output-file`` with the task
value.  This preserves the recipe idiom used by ``slurm.pl``:

.. code-block:: bash

   hyperion-submit slurm --cfg conf/submit_coe_v100.yaml \
     --num-gpus 1 --array JOB=1:100 \
     --output-file exp/xvector/log/extract.JOB.log -- \
     hyperion-extract-wav2xvectors --part-idx JOB --num-parts 100 \
       --output-spec ark,csv:exp/xvector/xvector.JOB.ark,exp/xvector/xvector.JOB.csv

For Slurm arrays, ``NAME`` is assigned from ``SLURM_ARRAY_TASK_ID`` in the
batch script.  For ``local`` arrays, tasks execute sequentially, in ascending
order, and the command stops at the first failed task.  Array tasks use
separate logs: the output-file must contain ``NAME`` or the launcher rejects
the invocation to prevent concurrent writes to one log.

Execution and failure semantics
-------------------------------

Both backends are synchronous by default:

* ``local`` returns only after the child process, or every local array task,
  finishes.
* ``slurm`` returns only after the submitted Slurm job or array has reached a
  terminal state.

The command exits zero only if every launched task exits zero.  Submission
errors, a missing log, a cancelled task, or a non-zero task exit status cause a
non-zero exit and identify the affected job/task logs.  The Slurm backend uses
Slurm accounting/state queries rather than completion sentinel files.

Slurm execution model
---------------------

The Slurm backend renders a temporary ``bash`` batch script and submits that
script with ``sbatch``.  The script is retained beside the submission log or
in a caller-visible temporary location when submission fails, so the exact
launched command can be inspected.  It uses argument-safe quoting rather than
building a shell command string from recipe text.

The requested ``--output-file`` is the program's combined log.  Slurm's own
fallback stdout and stderr files are stored in the adjacent ``q/`` directory:
``slurm-%j.out`` and ``slurm-%j.err`` for a regular job, or
``slurm-%A_%a.out`` and ``slurm-%A_%a.err`` for an array.  They normally only
contain scheduler or early script-launch diagnostics, and keep Slurm's default
``slurm-<jobid>.out`` files out of the submission directory.

The launcher requests resources from Slurm; it does not discover or select
GPUs itself.  In particular, it must not call ``free-gpu``.  The generated
script preserves Slurm's ``CUDA_VISIBLE_DEVICES`` value.  A missing value is
not treated as permission to select arbitrary GPUs.

If ``--num-gpus`` exceeds one, the generated command is prefixed with:

.. code-block:: bash

   torchrun --standalone --nnodes=1 --nproc-per-node=<num-gpus>

For zero or one GPU, the target command is invoked directly.  Distributed
Hyperion programs retain responsibility for their own distributed setup.

Runtime environment
-------------------

``hyperion-submit`` runs the target command in its own inherited environment;
it does not activate Conda, choose a Conda environment, or wrap the command in
``conda run``.  Thus a user activates the desired environment once before
launching the recipe, and both ``hyperion-submit`` and the final Hyperion
executable use that same environment.

For Slurm, the launcher always submits with ``--export=ALL``.  This transfers
the submitter's ``PATH``, Conda activation variables, library paths, and other
environment values to the batch job.  The batch script must not source Conda
initialization files or call ``conda activate``.  A cluster whose compute nodes
cannot execute the submitter's inherited environment is outside this initial
contract and must be fixed through its module/environment setup before running
the recipe.

YAML configuration
------------------

Each site YAML file is specific to the backend selected on the command line.
For example, ``hyperion-submit slurm --cfg conf/submit_coe_v100.yaml`` uses a
flat Slurm configuration; it does not repeat a redundant ``slurm:`` section:

.. code-block:: yaml

   num_gpus: 0
   num_threads: 1
   mem: 8G
   time: "168:00:00"
   sbatch_command: sbatch
   base_options:
     - --nodes=1
     - --ntasks-per-node=1
   cpu_options:
     - --partition=cpu
   default_gpu_type: v100
   gpu_types:
     v100:
       options:
         - --partition=gpu
         - --gres=gpu:v100:{num_gpus}

Top-level ``num_gpus``, ``num_threads``, ``mem``, ``mem_per_cpu``, and ``time``
are defaults for the matching command-line options. Command-line values
override the YAML values. The configuration may define at most one of ``mem``
and ``mem_per_cpu``. The Slurm backend translates these options as follows:

* ``num_threads``: ``--cpus-per-task=<value>``.
* ``mem``: ``--mem=<value>`` (memory per node; with the initial one-node
  contract, this is also the job total).
* ``mem_per_cpu``: ``--mem-per-cpu=<value>``.
* ``time``: ``--time=<value>``.
* zero GPUs: ``cpu_options``.
* one or more GPUs: the selected GPU type's options with ``{num_gpus}``
  expanded.

CPU and GPU policy selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the effective ``num_gpus`` is zero, the submitter appends
``cpu_options``. When it is greater than zero, it ignores ``cpu_options``,
looks up the name in ``default_gpu_type`` under ``gpu_types``, replaces
``{num_gpus}``, and appends that policy's ``options``.

For example, the following CLSP policy:

.. code-block:: yaml

   cpu_options:
     - --partition=gpu
     - --exclude=c04,c05,c23,c24,c25
   default_gpu_type: default
   gpu_types:
     default:
       options:
         - --partition=gpu
         - --gpus={num_gpus}
         - --exclude=c04,c05,c17,c20,c24,c25,c26,c27

produces ``--partition=gpu --exclude=c04,c05,c23,c24,c25`` for a CPU job. A
request containing ``--num-gpus 2`` instead produces
``--partition=gpu --gpus=2 --exclude=c04,c05,c17,c20,c24,c25,c26,c27``.

``base_options`` contains site-wide arguments passed directly to ``sbatch``.
It may include valid site policy arguments such as ``--nodes``, ``--account``,
``--qos``, or ``--constraint``. It must not override submitter-owned options:
``--export``, ``--parsable``, ``--output``, ``--error``, ``--array``,
``--cpus-per-task``, ``--mem``, ``--mem-per-cpu``, ``--time``, or GPU resource
selection. Values are individual argument strings, never shell fragments.

Slurm defines ``--mem``, ``--mem-per-cpu``, and ``--mem-per-gpu`` as mutually
exclusive resource requests.  The initial Hyperion interface exposes the
first two; a future ``--mem-per-gpu`` option can be added as a separately
specified extension. See the `Slurm sbatch documentation
<https://slurm.schedmd.com/sbatch.html>`_.

Only these documented placeholders are permitted in site options.  Arbitrary
shell fragments are not accepted from configuration files.

Initial non-goals
-----------------

The first release does not replace legacy wrappers outside the migrated
VoxCeleb top-level stages, does not support SGE, does not provide asynchronous
submission, does not activate or select Conda environments, and does not
expose arbitrary raw ``sbatch`` options.  These can be evaluated as later,
separately specified extensions.


# ---------------------------------------------------------------------------
# path.sh
# ---------------------------------------------------------------------------
# Minimal environment bootstrapper for the VoxCeleb v1.2 recipe.  Sourced by
# every stage to resolve relative paths and append the shared ``tools``
# directory to ``PATH``/``PYTHONPATH`` via the top-level script located at
# $TOOLS_ROOT/path.sh.  Modify the variables below if you check out the
# repository in a non-standard location.
# ---------------------------------------------------------------------------

# Root of the Hyperion repository (three directories above this file).
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
# Shared tools (Kaldi binaries, python utilities, etc.).
export TOOLS_ROOT=$HYP_ROOT/tools

# Source the global toolchain setup.
. $TOOLS_ROOT/path.sh

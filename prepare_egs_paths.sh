#!/bin/bash
# Copyright 2021 Johns Hopkins University  (Author: Jesus Villalba)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo "This prepare the paths to run the recipes in egs directory"

read -p "Introduce path to your conda base installation (e.g.:/usr/local/anaconda3): " CONDA_ROOT
read -p "Introduce name/prefix_path for your conda environment (e.g.:hyperion): " HYP_ENV

cat ./tools/proto_path.sh | \
  sed -e 's@__HYP_ENV__@'$HYP_ENV'@' \
  -e 's@__CONDA_ROOT__@'$CONDA_ROOT'@' \
  > ./tools/path.sh

# Check if Hyperion is installed in the environment
# if not add hyperion to python path
if [ -n "$CONDA_ROOT" ];then
    . $CONDA_ROOT/etc/profile.d/conda.sh
fi
conda activate $HYP_ENV
x=$(pip freeze | awk 'BEGIN{x=0} /hyperion/ { x=1 } END{ print x }')
if [ $x -eq 1 ];then
  echo "Hyperion is installed in env $HYP_ENV"
  echo "Recipes will use the installed one"
else
  echo "Hyperion is not installed in env $env"
  echo "Adding hyperion directory to the PYTHONPATH variable in the recipes"
  echo "export PYTHONPATH=\$HYP_ROOT:\$PYTHONPATH" >> ./tools/path.sh
fi 


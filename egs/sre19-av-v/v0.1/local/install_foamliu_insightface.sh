#!/bin/bash

echo "Download insightface repo"
# We use fork of repo at https://github.com/foamliu/InsightFace-PyTorch.git
# git clone
git clone https://github.com/jesus-villalba/InsightFace-PyTorch.git
cd InsightFace-PyTorch
#use sre21 branch
git checkout sre21-cpt
cd -

#!/bin/bash

# Install the IBM adversarial robustness toolbox from github

# To install from PyPi repo
# pip install adversarial-robustness-toolbox

# To install last version from github
git clone https://github.com/IBM/adversarial-robustness-toolbox.git

cd adversarial-robustness-toolbox
pip install .

#run tests
#so it doesn't fail the test you need to have tf, pytorch, keras installed in the env
bash run_tests.sh

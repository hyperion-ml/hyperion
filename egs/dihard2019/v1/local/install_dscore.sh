#!/bin/bash
# installs N. Ryan dscore tool
. path.sh
set -e

# we install it in the tools dir
if [ -f "$TOOLS_ROOT/dscore/score.py" ];then
    # The tool is already installed
    exit 0
fi

cd $TOOLS_ROOT
git clone https://github.com/jesus-villalba/dscore
cd -

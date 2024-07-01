#!/bin/bash

if test -f "/export/corpora6/ADI17/train_segments/QAT/V44GXdCBMPo_127038-12740999.wav"; then
    echo "File exists."
else
    echo "File does not exist."
fi

if [ -f "/export/corpora6/ADI17/train_segments/QAT/V44GXdCBMPo_127038-127406.wav" ]; then
    echo "File exists."
else
    echo "File does not exist."
fi

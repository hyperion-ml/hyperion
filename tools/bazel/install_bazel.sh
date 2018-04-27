#!/bin/bash

wget https://github.com/bazelbuild/bazel/releases/download/0.11.1/bazel-0.11.1-installer-linux-x86_64.sh

bash bazel-0.11.1-installer-linux-x86_64.sh --prefix=$HOME/usr/local/bazel

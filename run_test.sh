#!/bin/bash

rm -rf ./tests/data_out/*
mkdir ./tests/data_out/ark
mkdir ./tests/data_out/h5

py.test ./tests



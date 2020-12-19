#!/bin/bash

d=$1

awk '/PROG/{ printf "%.2f,%.3f,",$2,$3}' $d/sre19_eval_cmn2_results
awk '/EVAL/{ printf "%.2f,%.3f,",$2,$3}' $d/sre19_eval_cmn2_results
awk '{ printf "%.2f,%.3f,",$2,$4}' $d/sre16_eval40_yue_results
awk '{ printf "%.2f,%.3f\n",$2,$4}' $d/sre16_eval40_tgl_results

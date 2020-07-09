#!/bin/bash


run_vae.sh gmm GMM
run_vae.sh vent VENTILATOR

run_tied_vae.sh gmm_ivec_T1 "GMM IVEC T1D"
run_tied_vae.sh gmm_ivec_T10 "GMM IVEC"
run_tied_vae.sh vent_ivec "VENTILATOR IVEC"
run_tied_vae.sh vent_ivec_nonlin "VENTILATOR IVEC NONLIN"


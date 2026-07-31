# LRE22 Open Condition V1

Recipe for the NIST LRE22 open condition based to the JHU-MIT Submission.

## Citing
```
@inproceedings{villalba23_interspeech,
  author={Jesús Villalba and Jonas Borgstrom and Maliha Jahan and Saurabh Kataria and Leibny Paola Garcia and Pedro Torres-Carrasquillo and Najim Dehak},
  title={{Advances in Language Recognition in Low Resource African Languages: The JHU-MIT Submission for NIST LRE22}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={521--525},
  doi={10.21437/Interspeech.2023-1094}
}
```

## Training Data

  - x-Vector networks trained on:
    - VoxLingua107
    - NIST LRE17 Train + Dev + Eval / CTS + AfV  without Maghrebi Arabic
    - NIST SRE16
    - NIST SRE18
    - NIST SRE19 CMN2
    - NIST SRE21
    - NIST SRE CTS Superset
    - IARPA Babel
    - Fleurs
    - LWAZI 2009
    - NCHLT 2014
    - AMMI 2020
    - CommonVoice Tigrinya, Indian English, French
    - ADI 2017
    - AST
  - Gaussian back-end trained on:
    - NIST LRE22 dev with 2-fold cross-val + x10 augmentations

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it uses Res2Net50
   - To change the default network run scripts with the config-file argument:
```bash
run_011_train_xvector.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh
run_030_extract_xvectors.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh --use-gpu true
run_040_be_final.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh
```

## Results

| Config | Model Type | Model Details | Back-end | Dev MinCp | Dev ActCp | Eval MinCp | Eval ActCp |
| ------ | ---------- | ------------- | -------- | :-------: | :-------: | :--------: | :--------: |
| config_fbank64_stmn_ecapatdnn2048x4_v1.0.sh | ECAPA-TDNN 2048x4 | Stage-1 | GBE | 0.100 | 0.101 | 0.105 | 0.106 |
| config_fbank64_stmn_fwseres2net50s8_v1.0.sh  | fw-SE Res2Net50 scale=8 | Stage-1 | GBE | 0.092 | 0.093 | 0.103 | 0.104 |
| Fusion ECAPA-TDNN + FwSE Res2Net50 |  | | FoCal | 0.082 | 0.083 | 0.089 | 0.090 | 

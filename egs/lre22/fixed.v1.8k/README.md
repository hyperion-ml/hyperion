# LRE22 Fixed Condition V1

Recipe for the NIST LRE22 fixed condition based to the JHU-MIT Submission.

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
    - NIST LRE17 Train + Dev + Eval / CTS + AfV
  - Gaussian back-end trained on:
    - NIST LRE22 dev with 2-fold cross-val + x10 augmentations

## Usage

   - Run the run_0*.sh scripts in sequence
   - By default it uses ECAPA-TDNN 4 layers of 2048 dim.
   - To change the default network run scripts with the config-file argument:
```bash
run_011_train_xvector.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh
run_030_extract_xvectors.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh --use-gpu true
run_040_be_final.sh --config-file global_conf/config_fbank64_stmn_fwseres2net50s8_v1.0.sh
```

## Results

| Config | Model Type | Model Details | Back-end | Dev MinCp | Dev ActCp | Eval MinCp | Eval ActCp |
| ------ | ---------- | ------------- | -------- | :-------: | :-------: | :--------: | :--------: |
| config_fbank64_stmn_ecapatdnn2048x4_v1.0.sh | ECAPA-TDNN 2048x4 | Stage-2 | GBE | 0.207 | 0.209 | 0.198 | 0.199 |
| config_fbank64_stmn_fwseres2net50s8_v1.0.sh  | fw-SE Res2Net50 scale=8 | Stage-2 | GBE | 0.227 | 0.229 | 0.213 | 0.215 |
| Fusion ECAPA-TDNN + FwSE Res2Net50 |  | | FoCal | 0.182 | 0.183 | 0.180 | 0.181 |


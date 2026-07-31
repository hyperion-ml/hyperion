# HYPERION

<div align="center">

[![PyPI version](https://badge.fury.io/py/hyperion-ml.svg)](https://badge.fury.io/py/hyperion-ml)
[![License](https://img.shields.io/github/license/hyperion-ml/hyperion.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/hyperion-ml.svg)](https://pypi.org/project/hyperion-ml/)
[![Downloads](https://pepy.tech/badge/hyperion-ml)](https://pepy.tech/project/hyperion-ml)
[![Documentation Status](https://readthedocs.org/projects/hyperion-ml/badge/?version=latest)](https://hyperion-ml.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fhyperion-ml%2Fhyperion%2Fbadge&style=flat)](https://actions-badge.atrox.dev/hyperion-ml/hyperion/goto)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

Hyperion is a Speaker Recognition Toolkit based on PyTorch and numpy. It provides:
 - x-Vector architectures: ResNet, Res2Net, Spine2Net, ECAPA-TDNN, EfficientNet, Transformers and others.
 - Embedding preprocessing tools: PCA, LDA, NAP, Centering/Whitening, Length Normalization, CORAL
 - Several flavours of PLDA back-ends: Full-rank PLDA, Simplified PLDA, PLDA
 - Calibration and Fusion tools
 - Recipes for popular datasets: VoxCeleb, NIST-SRE, VOiCES

The full API is described in the documentation page [https://hyperion-ml.readthedocs.io](https://hyperion-ml.readthedocs.io)

## Building the documentation

Install the documentation dependencies and run the repository build script:

```bash
python3 -m pip install -r docs/requirements.txt
./docs/build.sh
```

The generated site is at `docs/_build/html/index.html`. The script treats
documentation warnings as errors. See the
[documentation build guide](docs/building-documentation.rst) for link checking,
doctests, and cleanup commands.

## Installation Instructions

### Prerequisites

    We use anaconda or miniconda, though you should be able to make it work in other python distributions
    To start, you should create a new enviroment:
```
conda create --name ${your_env} python=3.11
conda activate ${your_env}
```

### Installing Hyperion

- First, clone the repo:
```bash
git clone https://github.com/hyperion-ml/hyperion.git
```

- Then install hyperion in the environment, these are some valid commands depending pytorch and cuda versions:
```bash
cd hyperion
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -e .[torch29]
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .[torch29]
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .[torch29]
pip install --extra-index-url https://download.pytorch.org/whl/cu129 -e .[torch28]
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .[torch28]
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .[torch28]
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .[torch27]
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -e .[torch27]
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e .[torch26]
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .[torch25]
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .[torch24]
```

Known issues:

For older linux systems with GLIB <=2.17, try something like
```
pip install --extra-index-url https://download.pytorch.org/whl/cu121 -e .[torch25,gcc217] --only-binary=:all: --no-binary=intervaltree,fairscale
```

If you get this error when training:
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```
Unistall and reinstall numpy:
```
pip unistall numpy
pip install numpy=={same-version-you-uninstalled}
```

## Recipes

There are recipes for several tasks in the `./egs` directory.

### Prerequistes to run the recipes

These recipes require some extra tools (e.g. sph2pipe), which need to be installed first:
```bash
./install_egs_requirements.sh 
```

Most recipes do not require Kaldi, only the older ones using Kaldi x-vectors,
so we do not install it by default. If you are going to need it install it 
yourself. Then make a link in `./tools` to your kaldi installation
```bash
cd tools
ln -s ${your_kaldi_path} kaldi
cd -
```

Finally configure the python and environment name that you intend to use to run the recipes.
For that run
```bash
./prepare_egs_paths.sh
```
This script will ask for the path to your anaconda installation and enviromentment name.
It will also detect if hyperion is already installed in the environment,
otherwise it will add hyperion to your python path.
This will create the file
```
tools/path.sh
```
which sets all the enviroment variables required to run the recipes.
This has been tested only on JHU computer grids, so you may need to 
modify this file manually to adapt it to your grid.

## Recipes structure

The recipe layout is inspired by Kaldi: each dataset is prepared in a
directory under `data/`, and the numbered `run_*.sh` scripts operate on those
prepared datasets. However, the maintained recipes generally use Hyperion's
`HyperDataset` format rather than a collection of Kaldi mapping files. A
typical prepared dataset contains a manifest describing the available tables:
```
dataset.yaml
segments.csv       # segment IDs, recordings, speakers, timing, and metadata
recordings.csv     # recording IDs and audio storage paths
speaker.csv        # optional class/speaker information
trials.csv         # optional enrollment/test trial definitions
```

Some older recipes still use Kaldi-style data directories, with files such as
`wav.scp`, `utt2spk`, and `spk2utt`. Those layouts remain supported where the
recipe expects them, but they are not the standard output of the current
dataset-preparation classes.

### Running the recipes

Contrary to other toolkits, the recipes do not contain a single `run.sh` script 
to run all the steps of the recipe.
Since some recipes have many steps and most times you don't want to run all of then
from the beginning, we have split the recipe in several run scripts.
The scripts have a number indicating the order in the sequence.
For example,
```bash
run_001_prepare_data.sh
run_002_compute_vad.sh
run_010_prepare_audios_to_train_xvector.sh
run_011_train_xvector.sh
run_030_extract_xvectors.sh
run_040_evaluate_plda_backend.sh
```
will evaluate the recipe with the default configuration.
The default configuration is in the file `default_config.sh`

We also include extra configurations, which may change 
the hyperparamters of the recipe. For example:
 - Acoustic features
 - Type of the x-vector neural netwok
 - Hyper-parameters of the models
 - etc.

Extra configs are in the `global_conf` directory of the recipe.
Then you can run the recipe with the alternate config as:
```bash
run_001_prepare_data.sh --config-file global_conf/alternative_conf.sh
run_002_compute_vad.sh --config-file global_conf/alternative_conf.sh
run_010_prepare_audios_to_train_xvector.sh --config-file global_conf/alternative_conf.sh
run_011_train_xvector.sh --config-file global_conf/alternative_conf.sh
run_030_extract_xvectors.sh --config-file global_conf/alternative_conf.sh
run_040_evaluate_plda_backend.sh --config-file global_conf/alternative_conf.sh
```
Note that many alternative configus share hyperparameters with the default configs.
That means that you may not need to rerun all the steps to evaluate a new configuration.
It mast cases you just need to re-run the steps from the neural network training to the end.


## Citing

Each recipe README.md file contains the bibtex to the works that should be cited if you 
use that recipe in your research
     
## Directory structure:
 - The directory structure of the repo looks like this:
```bash
hyperion
hyperion/egs
hyperion/hyperion
hyperion/resources
hyperion/tests
hyperion/tools
```
 - Directories:
    - hyperion: python classes with utilities for speaker and language recognition
    - egs: recipes for sevaral tasks: VoxCeleb, SRE18/19/20, voices, ...
    - tools: contains external repos and tools like kaldi, python, cudnn, etc.
    - tests: unit tests for the classes in hyperion
    - resources: data files required by unittest or recipes

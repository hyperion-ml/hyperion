# HYPERION

Speech processing toolkit focused on speaker and language recognition

## Cloning the repo

- To clone the repo execute
```bash
git clone https://github.com/hyperion-ml/hyperion.git
```

## Dependencies:
  - Anaconda3.5:
     - Make a link to your anaconda installation in the tools directory:
     ```bash
     cd hyperion/tools/anaconda
     ln -s <your-anaconda-3.5> anaconda3.5
     ```
     - or follow instructions in hyperion/tools/anaconda/full_install.sh to install anaconda from scratch
  - Kaldi speech recognition toolkit
     - Make link to an existing kaldi installation
     ```bash
     cd hyperion/tools/kaldi
     ln -s <your-kaldi> kaldi
     ```
     - or follow instructions in hyperion/tools/anaconda/install_kaldi.sh to install kaldi from scratch

  - CuDNN: tensorflow and pytorch will need some version of cudnn
     - Make a link to some existing cudnn version that matches the requirements of your tf or pytorch, e.g.:
     ```bash
     cd hyperion/tools/cudnn
     #cudnn v7.4 for cuda 9.0 needed by pytorch 1.0 
     ln -s /home/janto/usr/local/cudnn-9.0-v7.4 cudnn-9.0-v7.4
     ```
  - In the CLSP grid: you can use some preinstalled versions of anaconda and kaldi in the grid to avoid each person having its own.
     - To create links to preinstalled kaldi, anaconda and cudnn, run:
     ```bash
     cd hyperion/
     ./make_clsp_links.sh
     ```
     - The anaconda that you will link with this has several environments:
        - base: numpy, h5py, pandas, etc.
	- tensorflow1.8g_cpu: tensorflow 1.8 for cpu
	- tensorflow1.8g_gpu: tensorflow 1.8 for gpu
	- pytorch1.0_cuda9.0: pytorch 1.0 with cuda 9.0
     
## Directory structure:
 - The directory structure of the repo looks like this:
```bash
hyperion
hyperion/egs
hyperion/hyperion
hyperion/resources
hyperion/tests
hyperion/tools
hyperion/tools/anaconda
hyperion/tools/cudnn
hyperion/tools/kaldi
hyperion/tools/keras
```
 - Directories:
    - hyperion: python classes with utilities for speaker and language recognition
    - egs: recipes for sevareal tasks: SRE18, voices, ...
    - tools: contains external repos and tools like kaldi, python, cudnn, etc.
    - tests: unit tests for the classes in hyperion
    - resources: data files required by unittest or recipes



# GIST-AI-HPC-2024
### GIST AI-HPC Tutorial

This repository contains the example code material for the GIST-AI-HPC-2024 tutorial:
*Deep Learning with HPC.

**Contents**
* [Links](#links)
* [Installation](#installation-and-setup)
* [Model, data, and code overview](#model-data-and-training-code-overview)
* [Single GPU training](#single-gpu-training)
* [Single GPU performance](#single-gpu-performance-profiling-and-optimization)
* [Distributed training](#distributed-gpu-training)
* [Multi GPU performance](#multi-gpu-performance-profiling-and-optimization)
* [Putting it all together](#putting-it-all-together)

## Links

Tutorial slides: https://drive.google.com/drive/folders/1rewCO1tVbE4rAat8VsCc-bKpNuEHUnNY?usp=sharing

Data download: Google Drive link

## Installation and Setup

### Software environment

Once logged into the MobileX, start a terminal and create a lab parent directory
```bash
mkdir Lab
cd Lab
```

To begin, start a terminal from JupyterHub and clone this repository with:
```bash
git clone https://github.com/BioHPC/GIST-AI-HPC-2024.git
```
In your terminal, change to the directory with
```bash
cd GIST-AI-HPC-2024
```

### Installing Nsight Systems (Skip this step if Nsight System is already installed)
In this tutorial, we will be generating profile files using NVIDIA Nsight Systems on thesystem. In order to open and view the
files on your local computer, you will need to install the Nsight Systems program, which you can download [here](https://developer.nvidia.com/gameworksdownload#?search=nsight%20systems). Select the download option required for your system (e.g. Mac OS host for MacOS, Window Host for Windows, or Linux Host .rpm/.deb/.run for Linux). You may need to sign up and create a login to NVIDIA's developer program if you do not
already have an account to access the download. Proceed to run and install the program using your selected installation method.

## Model, data, and training code overview

The model in this repository is adapted from a ResUNetPlusPlus application of deep learning ([Jha el al. 2019](https://ieeexplore.ieee.org/document/8959021), [GitHub](https://github.com/DebeshJha/ResUNetPlusPlus)) model for Colorectal Polyp Segmentation.


The U-Net model architecture is


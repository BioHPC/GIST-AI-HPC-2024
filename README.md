# GIST-AI-HPC-2024
### GIST AI-HPC Tutorial

This repository contains the example code material for the GIST-AI-HPC-2024 tutorial:
*Deep Learning with HPC.

**Contents**
* [Tutorial Materials](#tutorial-materials)
* [Installation and Setup Requirements](#installation-and-setup-requirements)
* [Model, data, and code overview](#model-data-and-training-code-overview)
* [Single GPU training](#single-gpu-training)
* [Single GPU performance and Optimization](#single-gpu-performance-profiling-and-optimization)
* [Distributed training](#distributed-gpu-training)
* [Multi GPU performance](#multi-gpu-performance-profiling-and-optimization)
* [Putting it all together](#putting-it-all-together)

## Tutorial Materials

Most of the code and resources are shared in this GitHub repository. 

[Tutorial slides Goole Drive](https://drive.google.com/drive/folders/1rewCO1tVbE4rAat8VsCc-bKpNuEHUnNY?usp=sharing)

[Data Google Drive Link](https://drive.google.com/drive/folders/1Ebs1zbAdwSWioZLMCorfn8Q5DLoChRYo?usp=sharing)

## Installation and Setup Requirements

### Software environment (Skip this step -- All requirements are pre-installed)

We have set up all requirements previously.

Many packages are required including
```
os
numpy
cv2
tensorflow
glob
tqdm
```

To install such requirments easily, you can create a conda environment with the provided "environment.yml"

```bash
conda env create -f environment.yml -n tf-gpu
conda activate tf-gpu
```

Also, you can check the requirments.txt if you would like to set-up venv environment without using conda. 

### Tutorial resources and data setup (You have to do)

Once logged into the MobileX, start a terminal and create a lab parent directory
```bash
mkdir Lab
cd Lab
```

To begin, start a terminal and clone this repository with:
```bash
git clone https://github.com/BioHPC/GIST-AI-HPC-2024.git
```
In your terminal, change to the directory with
```bash
cd GIST-AI-HPC-2024
```

Download the dataset (Kvasir-SEG.zip) from [Google Link](https://drive.google.com/drive/folders/1Ebs1zbAdwSWioZLMCorfn8Q5DLoChRYo?usp=sharing) and move the file into ~/Lab/GIST-AI-HPC-2024/data/ directory.

Decompress (unzip) it.
```bash
unzip Kvasir-SEG.zip
```

Kvasir-SEG dataset shoud has three directories: train, valid and test. 

### Installing Nsight Systems (Skip this step -- Nsight System is already installed)
In this tutorial, we will be generating profile files using NVIDIA Nsight Systems on thesystem. In order to open and view the
files on your local computer, you will need to install the Nsight Systems program, which you can download [here](https://developer.nvidia.com/gameworksdownload#?search=nsight%20systems). Select the download option required for your system (e.g. Mac OS host for MacOS, Window Host for Windows, or Linux Host .rpm/.deb/.run for Linux). 
You may need to sign up and create a login to NVIDIA's developer program if you do not already have an account to access the download. Proceed to run and install the program using your selected installation method.


## Model, data, and training code overview

The model in this repository is adapted from a ResUNetPlusPlus application of deep learning ([Jha el al. 2019](https://ieeexplore.ieee.org/document/8959021), [GitHub](https://github.com/DebeshJha/ResUNetPlusPlus)) model for Colorectal Polyp Segmentation. Instead of cloning the original git, let us use this updated codes for easy testing. 

We will test Kvasir-SEG data set ([paper](https://arxiv.org/abs/1911.07069), [data webpage](https://datasets.simula.no/kvasir-seg/)). Kvasir-SEG data set contains 1000 images that come from colonoscopy videos where the image size ranges from 332x487 to 1920x1072 pixels. Each image is pared with a mask which shows where the polp occurs. We split the data into training (880) and validation (120) sets. The split data is available to download from [Google Drive link](https://drive.google.com/drive/folders/1Ebs1zbAdwSWioZLMCorfn8Q5DLoChRYo?usp=sharing)

The diverse sizes of the images are corrected by a built in program which reduces the size of the images to a uniform 256x256 pixels.  This is done by cropping and resizing the images using cv2. 

Unet was developed for biomedical image segmentation and is used for problems related but not limited to image segmentation problems.  Unet model take input images and labeled masks for said images.  The name Unet describes the general shape of the model used to create the machine learning model.  This architecture consists of an encode which is responsible for extracting significant features from our input images.  The decoder section up samples intermediate features and constructing the final output.  These two sections are symmetrical in size and are connected to each other.  These connection help connect the extracted features of the encoder to the corresponding decoders features.  

ResUnetPlusPlus uses the Unet architecture as a base but adds residual elements to it.  ResUnetPlusPlus builds on the Unet architecture making it require more computational time to run.

## Single GPU training

To test the training step of ResUnetPlusPlus for testing with a smaller epoch size (3), you need to do the below step.
```bash
$ python runresnetplusplus_train.py --epochs=3
```
Again, you need to download the dataset into ./data folder which has train and valid directories with images and masks.

If you haven't received an error, let us run a default script (epochs = 100)

```bash
$ python runresnetplusplus_train.py 
```

## Single GPU performance and Optimization

### Using NVIDIA Nsight Systems to optimize deep learning on the GPU 

#### Overview

NVIDIA Nsight Systems is a comprehensive performance analysis tool that provides detailed insights into your application’s runtime behavior. By leveraging Nsight Systems, you can optimize deep learning workloads on the GPU, ensuring efficient resource utilization and improved performance. This tool is particularly useful for identifying bottlenecks, understanding GPU utilization, and optimizing multi-threaded CPU-GPU interactions.
##### Key Features
• System-Wide Performance Analysis: Nsight Systems offers a complete view of your system’s performance, encompassing CPU, GPU, OS runtime libraries, and more.
• Thread and Process Visualization: Gain insights into how threads and processes interact with each other and with the GPU, helping you to identify synchronization issues and optimize parallel execution.
• CUDA Activity Tracking: Monitor GPU activities, including memory transfers, kernel executions, and CUDA API calls, to understand and enhance GPU utilization.
• Profiler Overhead: Minimal overhead ensures that the profiling process does not significantly impact the performance of your application, allowing for accurate measurements.

#### Using NVIDIA Nsight Systems
1. Installation: Install NVIDIA Nsight Systems from the NVIDIA website or through package managers like apt or yum for Linux.
2. Profiling Your Application: Use the nsys command-line tool to profile your deep learning application. For example:
   bash
   Copy code
   nsys profile --trace=cuda,osrt --output=my_profile_report ./my_deep_learning_script.py
3. Analyzing the Report: Open the generated report file (.nsys-rep) in the Nsight Systems GUI. The GUI provides an intuitive timeline view and detailed breakdown of your application’s performance.
4. Identifying Bottlenecks: Examine the timeline for periods of inactivity or excessive synchronization waits. Look for warnings and messages that indicate potential issues.
5. Optimization: Based on the analysis, optimize your data pipeline, kernel launches, and thread synchronization to enhance overall performance.
#### Example: Profiling a TensorFlow Model
##### Here’s a basic example of how to profile a TensorFlow model training script using Nsight Systems:
1. Prepare your script: Ensure your deep learning script is ready for profiling. For example:
   python
   Copy code
   import tensorflow as tf
    ... (rest of your deep learning code)
   
   
2. Profile the script:
   nsys profile --trace=cuda,osrt --output=model_training_report python train_model.py
3. Analyze the Report:
    ◦ Open model_training_report.qdrep in the Nsight Systems GUI.
    ◦ Review the timeline to identify GPU activity, thread interactions, and any periods of idle time.
    ◦ Optimize your script based on the insights gained.

##### Example Report
![alt text](https://github.com/BioHPC/GIST-AI-HPC-2024/blob/main/nsys_training_report_screenshot.png)
    Processes and Threads: Multiple Python processes running with several threads each.
    CUDA Usage: Green and blue bars represent CUDA activities such as memory transfers and kernel executions.
    Thread Synchronization: Presence of pthread_cond_wait and futex indicating synchronization points.

      
#### By incorporating NVIDIA Nsight Systems into your optimization workflow, you can achieve significant improvements in the performance and efficiency of your deep learning models on the GPU. This tool is essential for developers aiming to leverage the full potential of GPU acceleration in their deep learning applications.




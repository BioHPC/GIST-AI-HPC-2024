# GIST-AI-HPC-2024
### GIST AI-HPC Tutorial

This repository contains the example code material for the GIST-AI-HPC-2024 tutorial:
*Deep Learning with HPC.

**Contents**
* [Tutorial Materials](#tutorial-materials)
* [Installation and Setup Requirements](#installation-and-setup-requirements)
* [Model, data, and code overview](#model-data-and-training-code-overview)
* [Single GPU training](#single-gpu-training)
* [Single GPU performance and Optimization](#single-gpu-performance-and-optimization)
* [Multi GPU training and performance](#multi-gpu-training-and-performance)
* [Putting it all together](#putting-it-all-together)

## Tutorial Materials

Most of the code and resources are shared in this GitHub repository. 

[Tutorial slides Goole Drive](https://drive.google.com/drive/folders/1rewCO1tVbE4rAat8VsCc-bKpNuEHUnNY?usp=sharing)

[Data Google Drive Link](https://drive.google.com/drive/folders/1Ebs1zbAdwSWioZLMCorfn8Q5DLoChRYo?usp=sharing)

## Installation and Setup Requirements

### Software environment (Skip this step -- All requirements are pre-installed)

We have set up all requirements previously. The main requirements are cudatoolkit=11.8, cudnn=8.8 (and 8.7), tensorflow 2.14 (2.13). 

Many packages are required including
```
os
numpy
cv2
tensorflow (2.14)
glob
tqdm
```

To install such requirements easily, you can create a conda environment with the provided "environment.yml"

```bash
conda env create -f environment.yml -n tf-gpu
conda activate tf-gpu
```

Also, you can check the requirements.txt if you would like to set-up a python venv environment without using conda. 

### Tutorial resources and data setup (You have to do)

Once logged into the MobileX, start a terminal. You have to activate pre-built conda environment "tf-gpu"
```bash
conda activate tf-gpu
```

Create a lab parent directory
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

Kvasir-SEG dataset should have three directories: train, valid, and test. 

### Installing Nsight Systems (Skip this step -- Nsight System is already installed)
In this tutorial, we will be generating profile files using NVIDIA Nsight Systems on the system. In order to open and view the
files on your local computer, you will need to install the Nsight Systems program, which you can download [here](https://developer.nvidia.com/gameworksdownload#?search=nsight%20systems). Select the download option required for your system (e.g. Mac OS host for MacOS, Windows host for Windows, or Linux Host .rpm/.deb/.run for Linux). 
You may need to sign up and create a login to NVIDIA's developer program if you do not already have an account to access the download. Proceed to run and install the program using your selected installation method.

### Additional Steps for Windows Users

1. **Update NVIDIA Drivers:**
   - Ensure your NVIDIA drivers are up-to-date. You can download the latest drivers from the [NVIDIA website](https://www.nvidia.com/Download/index.aspx).

2. **Check CUDA and cuDNN:**
   - Make sure CUDAToolkit (11.8 preferred) and cuDNN (8.7) are properly installed and are compatible with each other.
   - You can download CUDA [here](https://developer.nvidia.com/cuda-downloads).
   - You can download cuDNN [here](https://developer.nvidia.com/cudnn).
   - Add the paths to CUDA and cuDNN to your system variables.

3. **Verify GPU Access:**
   - Ensure that PyTorch and TensorFlow can properly access the GPU.

4. **Install Nsight Systems:**
   - Download and install Nsight Systems for Windows [here](https://developer.nvidia.com/gameworksdownload#?search=nsight%20systems).
   - Add the path to `nsys.exe` (Nsight Systems executable) to your system variables. This executable is usually located in the `target-windows-x64` or `host-windows-x64` directory within the Nsight Systems installation path.

By following these steps, you will be able to successfully install and use NVIDIA Nsight Systems on your local computer to view and analyze profile files.


## Model, data, and training code overview

The model in this repository is adapted from a ResUNetPlusPlus application of deep learning ([Jha el al. 2019](https://ieeexplore.ieee.org/document/8959021), [GitHub](https://github.com/DebeshJha/ResUNetPlusPlus)) model for Colorectal Polyp Segmentation. Instead of cloning the original git, let us use this updated codes for easy testing. 

We will test Kvasir-SEG data set ([paper](https://arxiv.org/abs/1911.07069), [data webpage](https://datasets.simula.no/kvasir-seg/)). Kvasir-SEG data set contains 1000 images that come from colonoscopy videos where the image size ranges from 332x487 to 1920x1072 pixels. Each image is paired with a mask which shows where the polyp occurs. We split the data into training (800),  validation (100), and test (100) sets. The split data is available to download from [Google Drive link](https://drive.google.com/drive/folders/1Ebs1zbAdwSWioZLMCorfn8Q5DLoChRYo?usp=sharing)

The diverse sizes of the images are corrected by a built-in program that reduces the size of the images to a uniform 256x256 pixels.  This is done by cropping and resizing the images using cv2. 

Unet was developed for biomedical image segmentation and is used for problems related but not limited to image segmentation problems.  Unet model takes input images and labeled masks for said images.  The name Unet describes the general shape of the model used to create the machine learning model.  This architecture consists of an encode which is responsible for extracting significant features from our input images.  The decoder section up samples intermediate features and constructing the final output.  These two sections are symmetrical in size and are connected to each other. These connections help connect the extracted features of the encoder to the corresponding decoder's features.  

ResUnetPlusPlus uses the Unet architecture as a base but adds residual elements to it.  ResUnetPlusPlus builds on the Unet architecture making it require more computational time to run.

## Single GPU training

To test the training step of ResUnetPlusPlus for testing with a smaller epoch size (3), you need to do the below step.
```bash
$ python runresnetplusplus_train.py --epochs=3
```
Again, you need to download the dataset into ./data folder which has train and valid directories with images and masks.

runresnetplusplus_train.py code train the model using GPU resource. Let us compare running time with CPU mode.

```bash
$ python runresnetplusplus_train_cpu.py --epochs=3
```
Quesiton: What is the difference in execution time between CPU and GPU modes?

If you haven't received an error, let us run a default script (epochs = 100)

```bash
python runresnetplusplus_train.py 
```

## Single GPU performance and Optimization

### Monitoring GPUs

#### The NVIDIA System Management Interface (nvidia-smi)
The basic check is
```bash
nvidia-smi
watch -n 1 nvidia-smi
```
The number after -n is the time interval, so if you type 1, you're telling it to update every second.

#### NVITop 
NVITop is a powerful monitoring tool designed specifically for NVIDIA GPUs. It is similar in concept to the traditional top command-line utility, but it is tailored to provide detailed information about the usage of NVIDIA GPUs on a system. NVITop is typically used in environments where multiple GPUs are employed, such as high-performance computing clusters, deep learning workstations, and data centers.

Install and run:
```bash
pip install nvitop 
nvitop
```
![alt text](https://github.com/BioHPC/GIST-AI-HPC-2024/blob/main/nvitop.png)


### Using NVIDIA Nsight Systems to optimize deep learning on the GPU 

#### Overview

NVIDIA Nsight Systems is a comprehensive performance analysis tool that provides detailed insights into your application’s runtime behavior. By leveraging Nsight Systems, you can optimize deep learning workloads on the GPU, ensuring efficient resource utilization and improved performance. This tool is particularly useful for identifying bottlenecks, understanding GPU utilization, and optimizing multi-threaded CPU-GPU interactions.
##### Key Features
- System-Wide Performance Analysis: Nsight Systems offers a complete view of your system’s performance, encompassing CPU, GPU, OS runtime libraries, and more.
- Thread and Process Visualization: Gain insights into how threads and processes interact with each other and with the GPU, helping you to identify synchronization issues and optimize parallel execution.
- CUDA Activity Tracking: Monitor GPU activities, including memory transfers, kernel executions, and CUDA API calls, to understand and enhance GPU utilization.
- Profiler Overhead: Minimal overhead ensures that the profiling process does not significantly impact the performance of your application, allowing for accurate measurements.

#### Using NVIDIA Nsight Systems
1. **Installation**: Install NVIDIA Nsight Systems from the NVIDIA website or through package managers like apt or yum for Linux.
2. **Profiling Your Application**: Use the nsys command-line tool to profile your deep learning application. For example:
```bash
nsys profile --trace=cuda,osrt --output=my_profile.nsys-rep python train.py
```
3. **Analyzing the Report**: Open the generated report file (.nsys-rep) in the Nsight Systems GUI. The GUI provides an intuitive timeline view and detailed breakdown of your application’s performance.
4. **Identifying Bottlenecks**: Examine the timeline for periods of inactivity or excessive synchronization waits. Look for warnings and messages that indicate potential issues.
5. **Optimization**: Based on the analysis, optimize your data pipeline, kernel launches, and thread synchronization to enhance overall performance.

#### Example: Profiling our TensorFlow lab model training
1. **Profile the script**:

```bash
nsys profile --trace=cuda,osrt --output=model_training.nsys-rep python runresnetplusplus_train.py
```
   - For Windows Users osrt is not a recognized trace argument. Instead, use the following command:
```bash
nsys profile --trace=cuda --output=model_training.nsys-rep python runresnetplusplus_train.py
```

2. **Analyze the Report**:
    - Open the report in the Nsight Systems GUI.
    - Review the timeline to identify GPU activity, thread interactions, and any periods of idle time.
    - Optimize your script based on the insights gained.

##### Example Report
![alt text](https://github.com/BioHPC/GIST-AI-HPC-2024/blob/main/nsys_training_report_screenshot.png)
    Processes and Threads: Multiple Python processes running with several threads each.
    CUDA Usage: Green and blue bars represent CUDA activities such as memory transfers and kernel executions.

      
##### By incorporating NVIDIA Nsight Systems into your optimization workflow, you can achieve significant improvements in the performance and efficiency of your deep learning models on the GPU. This tool is essential for developers aiming to leverage the full potential of GPU acceleration in their deep-learning applications.

##### Adding NVTX Annotations (Skip this step by limited time! Just explain using slides!)

NVTX ranges can be used to annotate:

large general code regions (training step/epoch, file I/O, etc) 

targeted code locations suspected of leading to GPU idle time

4.1_runresnetplusplusseg_train_nvtx.py and 4.2_runresnetplusplusseg_test_nvtx.py 
Added nvtx blocks for nsys profile

```bash
nsys profile -t nvtx,cuda -o nsys_report_4.1.nsys-rep python 4.1_runresnetplusplusseg_train.py --epochs 100
nsys profile -t nvtx,cuda -o nsys_report_4.2.nsys-rep python 4.2_runresnetplusplusseg_test.py
```

### DataLoader Change
The previous script used a custom data generator (DataGen), while the second script directly uses TensorFlow’s tf.data.Dataset API to load and preprocess the data. 

Please check the code

```base
5.1_runresnetplusplusseg_train_tfDataloader.py
```

Previous:
```bash
from data_generator import DataGen ... ## Generator nvtx.push_range("DataGen initialization") train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size) valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size) nvtx.pop_range()
```

Updated:
```bash
# Removed DataGen import
def parse_image(img_path, image_size):
    image_rgb = tf.io.read_file(img_path)
    image_rgb = tf.image.decode_jpeg(image_rgb, channels=3)
    image_rgb = tf.image.resize(image_rgb, [image_size, image_size])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.0
    return image_rgb

def parse_mask(mask_path, image_size):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_jpeg(mask, channels=1)
    mask = tf.image.resize(mask, [image_size, image_size])
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def load_data(image_paths, mask_paths, image_size):
    images = tf.data.Dataset.from_tensor_slices(image_paths)
    masks = tf.data.Dataset.from_tensor_slices(mask_paths)
    images = images.map(lambda x: parse_image(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    masks = masks.map(lambda x: parse_mask(x, image_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = tf.data.Dataset.zip((images, masks))
    return dataset

def prepare_dataset(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
...
    train_dataset = load_data(train_image_paths, train_mask_paths, image_size)
    valid_dataset = load_data(valid_image_paths, valid_mask_paths, image_size)

    train_dataset = prepare_dataset(train_dataset, batch_size)
    valid_dataset = prepare_dataset(valid_dataset, batch_size)
```

The tf.data.Dataset API is more integrated with TensorFlow and can be more efficient due to its optimizations (e.g., parallel data loading, prefetching).

The removal of DataGen simplifies the code and removes a dependency, making it more self-contained.

Previous:
```bash
model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks=callbacks)
```

Updated:
```bash
model.fit(train_dataset,
          validation_data=valid_dataset,
          epochs=epochs,
          callbacks=callbacks)
```

Impact:

- The previous script specifies steps_per_epoch and validation_steps, which are necessary for generators but not for tf.data.Dataset.
- The second script uses tf.data.Dataset objects (train_dataset and valid_dataset), which manage their own iteration without needing explicit steps per epoch.
- Using tf.data.Dataset can lead to more concise and potentially faster data handling, as TensorFlow can optimize the data pipeline more effectively.


## Multi GPU training and performance
Please refer slides

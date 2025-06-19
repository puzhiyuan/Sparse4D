FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV TZ Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},compute,display

SHELL [ "/bin/bash", "--login", "-c" ]

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

#install libs first
RUN apt-get update -q && \
    apt-get install -q -y \
    wget python3.8-dev python3-pip python3.8-tk git ninja-build \
    ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# intall pytorch
ENV TORCH_CUDA_ARCH_LIST "6.0 6.1 7.0 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS "-Xfatbin -compress-all"
ENV PATH ${PATH}:/usr/local/cuda:/usr/local/cuda/bin

# Install MMCV-series
ENV CUDA_HOME /usr/local/cuda
ENV FORCE_CUDA "1"

# Clone Sparse4D
RUN git clone https://github.com/puzhiyuan/Sparse4D.git
WORKDIR /Sparse4D

RUN pip install --upgrade pip
RUN pip install -r requirement.txt
RUN pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

# Link python to python3
RUN ln /usr/bin/python3 /usr/bin/python

# compile Sparse4D
WORKDIR /Sparse4D/projects/mmdet3d_plugin/ops
RUN python setup.py develop

WORKDIR /Sparse4D
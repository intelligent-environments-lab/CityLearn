FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER Bei Peng

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git

#Install python3 pip3
RUN apt-get update
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y python3.6 python3.6-dev
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN apt-get install -y python-apt --reinstall
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger

RUN mkdir /install
WORKDIR /install

# Install pymongo
RUN pip3 install pymongo

# install Sacred
RUN pip3 install setuptools
RUN git clone https://github.com/IDSIA/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

RUN pip3 install torch torchvision
RUN pip3 install snakeviz pytest probscale
RUN apt-get install -y htop iotop

#### -------------------------------------------------------------------
#### install mujoco
#### -------------------------------------------------------------------
RUN apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
RUN apt install -y unzip patchelf
RUN yes | pip3 uninstall enum34

ARG UID
RUN useradd -u $UID --create-home user
USER user
WORKDIR /home/user

RUN mkdir -p .mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d .mujoco \
    && rm mujoco.zip \
    && mv .mujoco/mujoco200_linux .mujoco/mujoco200

# Make sure you have a license
COPY ./mujoco_key.txt .mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /home/user/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

ENV MUJOCO_PY_MJKEY_PATH /home/user/.mujoco/mjkey.txt
ENV MUJOCO_PY_MUJOCO_PATH /home/user/.mujoco/mujoco200

RUN pip3 install --user mujoco-py
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco200/bin" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco200/bin" >> ~/.profile
RUN python3 -c "import mujoco_py"

# set python path
RUN echo "export PYTHONPATH=/home/user/pymarl" >> ~/.bashrc
RUN echo "export PYTHONPATH=/home/user/pymarl" >> ~/.profile

RUN pip3 install --user gym==0.10.8
EXPOSE 8888

WORKDIR /home/user/pymarl
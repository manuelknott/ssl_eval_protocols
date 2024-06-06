#!/bin/bash

# Install dependencies


sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 nano htop unzip  -y

pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install 'mmcv==2.0.0rc4'
pip install 'mmselfsup>=1.0.0rc0'


# create training scripts
#python3 create_training_scripts.py
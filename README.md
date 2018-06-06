# Egocentric Hand Detection 

## How to do training
1. Please select DeepQ-Synth-Hand-02 from training data provided by HTC
- Choose one dataset in DeepQ-Synth-Hand-02 folder to do training


2. quick run run_training.sh in HTC_training folder
- argv[1] is the path for images(/img) in training data
- argv[2] is the path for mask label(/mask) in training data

- Just run the following command:
```
$sh run_training.sh ./DeepQ-Synth-Hand-02/data/s005/img/ ./DeepQ-Synth-Hand-02/data/s005/mask/
```

3. The program loads model and do training

## How to do testing
1.  Download the judge_package: (supported by HTC.Taiwan)

Link: https://drive.google.com/open?id=1bDKe-lq3w6utonvZWDDOpWFMyhzYkswj

2. Follow the README.html in the package to install module judger_hand
```
$pip install judger_hand-0.3.0-py2.py3-none-any.whl
```
3. Run Egocentric_Hand_Detection/HTC_testing/run_testing.sh

4. The output should be the score of evaluation

## Environment
- Training data are from HTC Hand Detection provided by HTC.Taiwan. HTC hand detection module judger_hand should be installed.

- Compile on Ubuntu 16.04 platform and GPU workstation embedding Nvidia Telsa K40

- The following toolkits are used for our project
```
import sys

import numpy as np

import keras

from keras.layers import Concatenate,Add,Input,Dense,Activation,Conv2DTranspose,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Model, Sequential, load_model

from keras import backend as K

import pickle

import os

import matplotlib.image as mtimg

import matplotlib.pyplot as plt

import random as rnd

from PIL import Image

import cv2 as cv

import math

import random

from random import shuffle

import judger_hand
```








HTC & MLDS Competition 2017
Hand Detection

#######如何跑Training#########
1. 請先選擇HTC所提供的training data的包裝DeepQ-Synth-Hand-02當中的其中一個dataset作training

2. 跑HTC_training資料夾當中的run_training.sh 
sys.argv[1]是training data當中的img資料夾中的圖片
sys.argv[2]是training data當中的mask資料夾中的圖片

我是執行以下指令:
$sh run_training.sh ./DeepQ-Synth-Hand-02/data/s005/img/ ./DeepQ-Synth-Hand-02/data/s005/mask/

3. 程式會load model並開始做training



#######如何跑Testing##########
1. 到Facebook MLDS社團至頂文章當中下載HTC所提供的judge_package:
也可以從此連結下載: https://drive.google.com/open?id=1bDKe-lq3w6utonvZWDDOpWFMyhzYkswj

2. 依package中的README.html安裝module judger_hand
$pip install judger_hand-0.3.0-py2.py3-none-any.whl

3. 執行我們github專案MLDS_Final_2017當中的HTC_testing裡面有個run_testing.sh

4. 輸出應為HTC judge 出來的分數。



#######實驗環境描述（所需資料、系統、所需所有套件版本等）##########

主要會需要的資料會是HTC Hand detection 所給的training data。並且安裝HTC hand detection module judger_hand
我們是在Ubuntu 16.04上面運用Nvidia Telsa k40做執行。


以下為我們使用的的套件（版本皆為最新版本）

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










# coding: utf-8

# In[1]:

import sys
import numpy as np
import keras
from keras.layers import Concatenate,Add,Input,Dense,Activation,Conv2DTranspose,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model
from keras import backend as K
import pickle
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import matplotlib.image as mtimg
import matplotlib.pyplot as plt
import random as rnd
from PIL import Image
import cv2 as cv
import math
import random
from random import shuffle




training_img_path = sys.argv[1] #'./Dataset/DeepQ-Synth-Hand-02/data/s005/img/'
training_mask_path = sys.argv[2] #'./Dataset/DeepQ-Synth-Hand-02/data/s005/mask/'


# In[2]:


def get_the_list_name(path):
    img_name_list = []

    for filename in os.listdir(path):
        img_name_list.append(filename)
    
    if img_name_list[0][:3] == 'img':
        start = 'img_'
        end = '.png'
    if img_name_list[0][:3] == 'lab':
        start = 'label_'
        end = '.json'
    if img_name_list[0][:3] == 'mas':
        start = 'mask_'
        end = '.png'
    new_list=[]
    for i in range(len(img_name_list)):
        num = str(i)
        while len(num) < 8:
            num = '0'+num
        new_list.append(start+num+end)
        
    return new_list

def shuffle_two(a,b):
    
    c = list(zip(a, b))
    random.shuffle(c)
    x, y = zip(*c)
    return x,y

def img_2_array(img_name):
    
    img = Image.open(img_name)
    img = img.convert('RGB')
    img = np.array(img)
    return img

def plot_img(image):
    plt.imshow(image)
    plt.show()


# In[3]:


def fcn_label_maker(input_mask_list):
    
    return_label_imgs = []
   
    for i in range(len(input_mask_list)):

        label_img = np.zeros((240,320,3))
            
        for k in range(240):
            for j in range(320):
                color = input_mask_list[i][k][j]
                if color[0] == 255:
                    label_img[k][j][1] = 1
                    continue
                if color[1] == 255:
                    label_img[k][j][2] = 1
                    continue
                else: 
                    label_img[k][j][0] = 1
                
        return_label_imgs.append(label_img)

    return return_label_imgs


# In[4]:


def data_prepare(raw_img_path,
                 mask_img_path):
    
    raw_img_names = get_the_list_name(raw_img_path)
    mask_img_names = get_the_list_name(mask_img_path)
    
    shuffled = shuffle_two(raw_img_names,mask_img_names)
    
    return shuffled[0],shuffled[1]
    


# In[5]:


def fcn_data_producer(raw_img_path,
                      mask_img_path,
                      batch_size
                      ):
    i = 0
    
    names = data_prepare(raw_img_path,mask_img_path)
    raw_img_name_list = list(names[0])
    mask_img_name_list = list(names[1])
    
    
    
    while True:
        input_x = raw_img_name_list[(i*batch_size):(i*batch_size+batch_size)]
        label_y = mask_img_name_list[(i*batch_size):(i*batch_size+batch_size)]
        
        for i in range(batch_size):
            input_x[i] = img_2_array(raw_img_path+input_x[i])
            label_y[i] = img_2_array(mask_img_path+label_y[i])
                
            
        input_x = np.array(input_x)
        input_x = input_x.astype('float')
        input_x = input_x/255
            
        label_y = fcn_label_maker(label_y)
        label_y = np.array(label_y)
        
        i+=1

	#print(input_x.shape)	
	#print(label_y.shape)
        
        yield input_x,label_y


# In[6]:

#FCN_modelv2 = 1e_s002, 1e_s000, 1e_s001, 1e_s003, 1e_s004
#FCN_modelv3 = 2epoch for s000-s004
#FCN_modelv30114 = 2 epochs for s000-s004

#FCN_modelv4 = 4epoch for s000-004 
#FCN_modelv6 = modelv4+s005-008 for 2 epo



model = load_model('./training_model/FCN_modelv3vv.h5')

model.compile(loss='categorical_crossentropy',
              optimizer = keras.optimizers.Adam(lr=0.0005),metrics=['accuracy'])

model.fit_generator(fcn_data_producer(training_img_path,#img
                                      training_mask_path,#mask
                                       10),#batch_size
                    steps_per_epoch=999, 
                    epochs=2, 
                    verbose=1)

model.save('./training_model/FCN_modelv3vv.h5')

del model
K.clear_session()



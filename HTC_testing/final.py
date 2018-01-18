
# coding: utf-8

# In[4]:

import os
import numpy as np


from PIL import Image
#import cv2 as cv
import math


from keras import backend as K


import keras
from keras.layers import Concatenate,Add,Input,Dense,Activation,Reshape,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential, load_model

import judger_hand
# In[5]:

def max_idx(input_array):
    return list(input_array).index(max(list(input_array)))

def really_label(input_label):

    new_label = [input_label[2],input_label[0],input_label[3],input_label[1]]
    new_label[0] = int(new_label[0]*1.9125)
    new_label[2] = int(new_label[2]*1.9125)
    new_label[1] = int(new_label[1]*1.9167)
    new_label[3] = int(new_label[3]*1.9167)
    return (new_label[0],new_label[1],new_label[2],new_label[3])



def img_sample(input_img, sample_pixel, sample_size, draw_img='True'):

    sample_points = []
    
    for i in range(input_img.shape[0]):
        for k in range(input_img.shape[1]):
            if (sample_pixel == input_img[i][k]).all()==True:
                
                min_distance = 100000
                
                for j in range(len(sample_points)):
                    
                    distance_y = abs(i-sample_points[j][0]) 
                    distance_x = abs(k-sample_points[j][1])
                    
                    distance = math.sqrt(pow(distance_y,2) + pow(distance_x,2))
                    
                    if distance < min_distance:
                        min_distance = distance
                
                if min_distance > sample_size:
                    sample_points.append((i,k))
    
    if draw_img == 'True':
        output = np.zeros((test_img.shape[0],test_img.shape[1],3))
        output = output.astype('uint8')

        for i in range(test_img.shape[0]):
            for k in range(test_img.shape[1]):
                if (i,k) in sample_points:
                    output[i][k] = np.array([0,0,255])
                else:
                    output[i][k] = test_img[i][k]
        plot_img(output)
    
    return sample_points
def slice_the_img(input_img, img_range):
    img_h = img_range[1] - img_range[0]
    img_w = img_range[3] - img_range[2]
    
    slice_img = np.zeros((img_h,img_w,3))
    slice_img = slice_img.astype('uint8')
    
    for i in range(img_h):
        for k in range(img_w):

            slice_img[i][k] = input_img[img_range[0]+i][img_range[2]+k]
#     plot_img(slice_img)
    return slice_img

def img_2_array(img_name):
    
    img = Image.open(img_name)
    img = img.convert('RGB')
    img = np.array(img)
    return img

def img_reshape(input_img, target_shape):
    input_img = Image.fromarray(input_img.astype('uint8'))
    input_img = input_img.resize(target_shape, Image.BILINEAR)
    input_img = input_img.convert('RGB')
    input_img = np.array(input_img)
    
    return input_img

def get_the_list_name(path):
    img_name_list = []

    for filename in os.listdir(path):
        img_name_list.append(filename)
    
#     if img_name_list[0][:3] == 'img':
#         start = 'img_'
#         end = '.png'
#     if img_name_list[0][:3] == 'lab':
#         start = 'label_'
#         end = '.json'
#     if img_name_list[0][:3] == 'mas':
#         start = 'mask_'
#         end = '.png'
#     new_list=[]
#     for i in range(len(img_name_list)):
#         num = str(i)
#         while len(num) < 8:
#             num = '0'+num
#         new_list.append(start+num+end)
        
    return img_name_list

def result_to_mask(input_results):
    
    mask_results = []
    
    for i in range(len(input_results)):
        mask_img = np.zeros((240,320,3))
        
        for k in range(240):
            for j in range(320):
                idx = list(input_results[i][k][j]).index(max(list(input_results[i][k][j])))
                if idx == 0:
                    mask_img[k][j] = np.array([0,0,0])
                    continue
                if idx == 1:
                    mask_img[k][j] = np.array([255,0,0])
                    continue
                if idx == 2:
                    mask_img[k][j] = np.array([0,255,0])
        mask_img = mask_img.astype('uint8')
        mask_results.append(mask_img)
        
    return mask_results
        
def get_proposal_box(input_img, box_center, box_size, draw_img='True'):

    box_h = box_size[0]
    box_w = box_size[1]
    
    img_h = input_img.shape[0]
    img_w = input_img.shape[1]
    
    i = box_center[0]
    k = box_center[1]
    
    bounding = [i+int(box_h/2), i-int(box_h/2), k+int(box_w/2), k-int(box_w/2)]  #ideal box range
                
                
    if bounding[0] > img_h:    #if out of range
         bounding[0] = img_h
    if bounding[1] < 0:
         bounding[1] = 0
            
    if bounding[2] > img_w:
         bounding[2] = img_w
    if bounding[3] < 0:
         bounding[3] = 0
    
    #save in list
    bounding_height = [bounding[1],bounding[0]]  # [h_min , h_max]
    bounding_width = [bounding[3],bounding[2]]   # [w_min, w_max]  
    
    if draw_img == 'True':
        
        bounding_height+=[bounding[1]+1,bounding[1]+2,bounding[0]-1,bounding[0]-2]
        bounding_width +=[bounding[3]+1,bounding[3]+2,bounding[2]-1,bounding[2]-2]    
    
        
        box_color = np.array([33,33,255])        #box color
        box_color = box_color.astype('uint8')
    
        return_img = np.zeros((img_h,img_w,3))   
        return_img = return_img.astype('uint8')
    
        for i in range(img_h):
            for k in range(img_w):
                if (i > bounding_height[0] and i < bounding_height[1]) and (k in bounding_width):
                    return_img[i][k] = box_color

                elif (k > bounding_width[0] and k < bounding_width[1]) and (i in bounding_height):
                    return_img[i][k] = box_color

                else:
                    return_img[i][k] = input_img[i][k]

        return return_img,(bounding[1],bounding[0],bounding[3],bounding[2])
    else:
        return (bounding[1],bounding[0],bounding[3],bounding[2])
    
def pakage_raw_imgs(raw_imgs,shape):
    input_x = []
    
    for i in range(len(raw_imgs)):
        img = img_reshape(raw_imgs[i],shape)
        input_x.append(img)
    input_x = np.array(input_x)
    input_x = input_x.astype('float')
    input_x = input_x/255
    
    return input_x

#def results_report(input_results, idx):
#    prob = -1
#    number = -1
#    
#    for i in range(len(input_results)):
#        pos = input_results[i][idx]
#        obj = max_idx(input_results[i])
#        
#        if pos > prob and obj == idx:
#            prob = pos
#            number = i
#
#    return number,prob

def results_report(input_results, idx):
    
    prob_list = []
    number_list = []
    
    for i in range(len(input_results)):
        pos = input_results[i][idx]

        prob_list.append(pos)
        number_list.append(i)
        
        
    prob = max(prob_list)
    number = number_list[max_idx(prob_list)]
    
    return number,prob


# In[11]:

def test_fot_one(img_path,
                 fcn_model,
                 boxes_style,
                 predict_shape,
                 class_model, count
                 ):
    raw_img = img_2_array(img_path)
    raw_img = img_reshape(raw_img,(320,240))
    raw_img = raw_img.astype('float')
    raw_img = raw_img/255
    raw_img = np.reshape(raw_img,(1,240,320,3))
    
    fcn_model = load_model(fcn_model)
    
    mask_img = [fcn_model.predict(x=raw_img)[0]]
    mask_img = result_to_mask(mask_img)
    mask_img = mask_img[0]
    
    raw_img = img_2_array(img_path)
    raw_img = img_reshape(raw_img,(320,240))
    
    red_samples = img_sample(mask_img, np.array([255,0,0]), 8, draw_img='False')
    green_samples = img_sample(mask_img, np.array([0,255,0]), 8, draw_img='False')
    
    class_model = load_model(class_model)
    
    if len(red_samples) > 0:
        red_slice = []
        red_proposal_box = []

        for i in range(len(boxes_style)):
            box = boxes_style[i]        
            for k in range(len(red_samples)):
                proposal_box = get_proposal_box(raw_img, red_samples[k], box, draw_img='False')
                red_proposal_box.append(proposal_box)
                sliced_img = slice_the_img(raw_img, proposal_box)
                red_slice.append(sliced_img)
        red_slice = pakage_raw_imgs(red_slice, predict_shape)
        red_results = class_model.predict(x=red_slice)
        red_results = results_report(red_results,1)
        red_box = really_label(red_proposal_box[red_results[0]])
        red_posb = red_results[1]
        
        red_return = [red_box,red_posb]
 
    else: 
        red_return = [-1,-1]
    

    
    if len(green_samples) >0:
    
        green_slice = []
        green_proposal_box = []
    
        for i in range(len(boxes_style)):
            box = boxes_style[i]
        
            for j in range(len(green_samples)):
                proposal_box = get_proposal_box(raw_img, green_samples[j], box, draw_img='False')
                green_proposal_box.append(proposal_box)
                sliced_img = slice_the_img(raw_img, proposal_box)
                green_slice.append(sliced_img)
    

        green_slice = pakage_raw_imgs(green_slice, predict_shape)
        green_results = class_model.predict(x=green_slice)
        green_results = results_report(green_results,2)

        green_box = really_label(green_proposal_box[green_results[0]])
        green_posb = green_results[1]

        
        green_return = [green_box,green_posb]

    else: 
        green_return = [-1,-1]
    
    
    if red_return[1] ==-1:red_return[0]=-1
    if green_return[1] ==-1:green_return[0]=-1
    
    
    #red_return[0] : 左手框(x0, y0, x1, y1)
    #red_return[1] : 左手機率
    #green_return[0] : 右手框(x0, y0, x1, y1 )
    #green_return[1] : 右手機率
    #都已經轉換完成是答案
    #如果沒有偵測到那隻手 就會是 -1跟-1

   
    raw_img = img_2_array(img_path)
    
#    print('找到左手,機率為:'+str(red_results[1])+',位置為'+str(really_label(red_proposal_box[red_results[0]])))
#    ##    plot_img(red_img)
#    print('找到右手,機率為:'+str(green_results[1])+',位置為'+str(really_label(green_proposal_box[green_results[0]])))
 
    
    if red_return[1]>green_return[1] and green_return[1] < 0.9:
        green_return = [-1,-1]
    if green_return[1]>red_return[1] and red_return[1] < 0.9:
        red_return = [-1,-1]
#   
#    print(red_return,green_return)
    left = red_return[0]
    right = green_return[0]
    

    
#    if left != -1:
#        raw_img = cv.rectangle(raw_img, (left[0],left[1]), (left[2],left[3]), (255,255,255), 2) 
#    if right != -1:
#        raw_img = cv.rectangle(raw_img, (right[0],right[1]), (right[2],right[3]), (100,100,100), 2) 
#    
#    cv.imwrite(str(count)+'_bbox4.jpg', raw_img)
    
        
    if left !=-1:
        predict_info_l = img_path+' '+str(left[0])+' '+str(left[1])+' '+str(left[2])+' '+str(left[3])+' '+'0'+' '+str(red_return[1])
    else:
        predict_info_l = '0'
    if right !=-1:
        predict_info_r = img_path+' '+str(right[0])+' '+str(right[1])+' '+str(right[2])+' '+str(right[3])+' '+'1'+' '+str(green_return[1])
    else:
        predict_info_r = '0'
    
    return predict_info_l, predict_info_r
#    plot_img(green_img)
    
    


# In[ ]:
file_name = judger_hand.get_file_names()


predict_list = []


count = 0    
for file in file_name:    
    pre_l, pre_r = test_fot_one(file,
            './testing_model/FCN_modelv3vv.h5',
             [(50,60),(80,90),(125,135),(145,155),(170,180)],
             (80,80),
             './testing_model/fine_tune_class_model.h5',count)
    print(count)
    K.clear_session()
    
    count+=1
    if pre_l!='0':
        predict_list.append(pre_l)
    if pre_r!='0':
        predict_list.append(pre_r)

f = judger_hand.get_output_file_object()
for lines in predict_list:
    f.write((lines+'\n').encode('ascii'))



score, err = judger_hand.judge()

#name_list = get_the_list_name('HTCFILE/')
#
#for s in range(len(name_list)):
#
#
#    test_fot_one('HTCFILE/'+name_list[s],
#            'FCN_modelv3v5.h5',
#             [(40,40),(60,60),(100,100),(140,140)],
#             (80,80),
#             'class_model_9413.h5')


# In[ ]:




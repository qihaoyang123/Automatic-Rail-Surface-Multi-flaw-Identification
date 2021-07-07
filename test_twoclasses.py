# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import KFold
import cv2
import os
import time
from sklearn.model_selection import train_test_split
from custom_layers.scale_layer import Scale
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def class_weighted_crossentropy(target, output):
    axis = -1
    output /= tf.reduce_sum(output, axis, True)
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.225, 0.225, 0.1, 0.225, 0.225]
    return -tf.reduce_sum(target * weights * tf.log(output))

if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 2
    batch_size = 16
    nb_epoch = 100
    file = pd.read_csv('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/analysis_validation_select_checked.csv')
    label_dict = {'normal':0,'others':1}
    x = []
    y = np.array([]) 
    def read_image(path,label_name):
        img_path = os.listdir(path)
        data = []
        y = np.array([])
        for i in img_path:
            img = cv2.imread(os.path.join(path,i))
            if img.shape[1]<224:
                img = cv2.resize(cv2.copyMakeBorder(img,0,0,int((224-img.shape[1])/2),int((224-img.shape[1])/2),cv2.BORDER_CONSTANT,value=255),(224,224))
            else: img = cv2.resize(img,(224,224))
            data.append(img)
            y = np.append(y,label_dict[label_name])
        return np.array(data),y
    x1_,y1_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/corrugation_new','others')
    x1,y1 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Corrugation','others')
    x2,y2 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Defect','others')
    x3,y3 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Rail_with_Grinding_Mark','others')
    x4,y4 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Shelling','others')
    x4_,y4_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/shelling_new','others')
    x5,y5 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Squat','others')
    x5_,y5_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/squat_new','others')
    x6,y6 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/normal_all_resize','normal')
    kf = KFold(n_splits=4)
    for threshold in range(1,2):
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=20)
        x1_train_, x1_test_, y1_train_, y1_test_ = train_test_split(x1_, y1_, test_size=0.3, random_state=20)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=20)
        x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=20)
        x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.3, random_state=20)
        x4_train_, x4_test_, y4_train_, y4_test_ = train_test_split(x4_, y4_, test_size=0.3, random_state=20)
        x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.3, random_state=20)
        x5_train_, x5_test_, y5_train_, y5_test_ = train_test_split(x5_, y5_, test_size=0.3, random_state=20)
        x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y6, test_size=0.3, random_state=20)
        x_train = np.concatenate((x1_train,x2_train,x3_train,x4_train,x5_train,x6_train,x1_train_,x4_train_,x5_train_,x6_train))
        x_test = np.concatenate((x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x1_test_,x4_test_,x5_test_,x6_test))
        y_train = np.concatenate((y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y1_train_,y4_train_,y5_train_,y6_train))
        y_test = np.concatenate((y1_test,y2_test,y3_test,y4_test,y5_test,y6_test,y1_test_,y4_test_,y5_test_,y6_test))
        y_train = np_utils.to_categorical(y_train,num_classes=2)
        y_test = np_utils.to_categorical(y_test,num_classes=2)
        model = load_model(filepath = '/home/daisy001/qihaoyang/cnn_finetune-master_/model/densenet_169_06-29_all_twoclasses',custom_objects={'Scale':Scale})
        time_start=time.time()
        pre = model.predict(x_test)
        time_end=time.time()
        print('totally cost',(time_end-time_start)/len(y_test))
        pre = np.argmax(pre,axis=1)
        y_test = np.argmax(y_test,axis=1)
        score=accuracy_score(y_test,pre)
        print(score)
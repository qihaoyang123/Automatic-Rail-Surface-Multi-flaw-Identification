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
from sklearn.model_selection import train_test_split
import cv2
import os
from custom_layers.scale_layer import Scale
import time
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def class_weighted_crossentropy(target, output):
    axis = -1
    output /= tf.reduce_sum(output, axis, True)
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.5, 0.5, 0.3, 0.6, 0.7]
    return -tf.reduce_sum(target * weights * tf.log(output))

if __name__ == '__main__':

    img_rows, img_cols = 299, 299 # Resolution of inputs
    channel = 3
    num_classes = 3
    batch_size = 8
    nb_epoch = 100
    label_dict = {'Corrugation':0,'Defect':1,'Squat':2}
    x = []
    y = np.array([])

    def read_image(path,label_name):
        img_path = os.listdir(path)
        data = []
        y = np.array([])
        for i in img_path:
            img = cv2.imread(os.path.join(path,i))
            if img.shape[1]<299:
                img = cv2.resize(cv2.copyMakeBorder(img,0,0,int((299-img.shape[1])/2),int((299-img.shape[1])/2),cv2.BORDER_CONSTANT,value=255),(299,299))
            else: img = cv2.resize(img,(299,299))
            data.append(img) 
            y = np.append(y,label_dict[label_name])
        return np.array(data),y
    kf = KFold(n_splits=4)
    x2_,y2_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/defect_online','Defect')
    x1_o,y1_o = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Online Data/corrugation_new_online','Corrugation')
    x5_o,y5_o = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Online Data/squat_new_online','Squat')
    score = 0
    score1 = 0
    score2 = 0
    score3 = 0
    for threshold in range(1,2):
        x2_train_, x2_test_, y2_train_, y2_test_ = train_test_split(x2_, y2_, test_size=0.9, random_state=20)
        x1_o_train, x1_o_test, y1_o_train, y1_o_test = train_test_split(x1_o, y1_o, test_size=0.9, random_state=20)
        x5_o_train, x5_o_test, y5_o_train, y5_o_test = train_test_split(x5_o, y5_o, test_size=0.9, random_state=20)
        def augmentation(x1_train,y1_train):
            x1_train_ = x1_train
            x1_train1 = x1_train
            x1_train2 = x1_train
            x1_train3 = x1_train
            y1_train_ = y1_train
            y1_train1 = y1_train
            y1_train2 = y1_train
            y1_train3 = y1_train
            for i in range(0,len(x1_train)):
                x1_train1[i] = cv2.flip(x1_train[i],1)
            for i in range(0,len(x1_train)):
                x1_train2[i] = cv2.flip(x1_train[i],0)
            for i in range(0,len(x1_train)):
                x1_train3[i] = cv2.flip(x1_train[i],-1)
            y1_train = np.concatenate((y1_train_,y1_train1,y1_train2,y1_train3))
            x1_train = np.concatenate((x1_train_,x1_train1,x1_train2,x1_train3))
            return x1_train, y1_train
        x2_train_,y2_train_ = augmentation(x2_train_,y2_train_)
        x1_o_train,y1_o_train = augmentation(x1_o_train,y1_o_train)
        x5_o_train,y5_o_train = augmentation(x5_o_train,y5_o_train)     
        x_train = np.concatenate((x2_train_,x1_o_train,x5_o_train))
        x_test = np.concatenate((x2_test_,x1_o_test,x5_o_test))
        #y_train = np.concatenate((y2_train_,y1_o_train,y5_o_train,y3_train,y4_train))
        #y_test = np.concatenate((y2_test_,y1_o_test,y5_o_test,y3_test,y4_test))
        y_train = np.concatenate((y2_train_,y1_o_train,y5_o_train))
        y_test = np.concatenate((y2_test_,y1_o_test,y5_o_test))
        y_train = np_utils.to_categorical(y_train,num_classes=3)
        y_test = np_utils.to_categorical(y_test,num_classes=3)
        model = load_model(filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/inceptionv4_03-06_all_online',custom_objects={'Scale':Scale})
        time_start=time.time()
        pre = model.predict(x_test)
        pre = np.argmax(pre,axis=1)
        y_test = np.argmax(y_test,axis=1)
        time_end=time.time()
        print('totally cost',(time_end-time_start)/len(y_test))
        score+=accuracy_score(y_test,pre)
        predictions_valid = model.predict(x2_test_)
        prediction = np.argmax(predictions_valid,axis=1)
        score1+=accuracy_score(y2_test_, prediction)
        predictions_valid = model.predict(x1_o_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score2+=accuracy_score(y1_o_test, prediction)
        predictions_valid = model.predict(x5_o_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score3+=accuracy_score(y5_o_test, prediction)      
        print(score,score1,score2,score3)
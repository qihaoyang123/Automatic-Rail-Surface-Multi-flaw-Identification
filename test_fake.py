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

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 5
    batch_size = 8
    nb_epoch = 100
    label_dict = {'normal':0,'Corrugation':1,'Defect':2,'Rail with Grinding Mark':3,'Shelling':4,'Squat':5}
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
    kf = KFold(n_splits=4)
    x1_f,y1_f = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/fake_images/corrugation','Corrugation')
    x2_f,y2_f = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/fake_images/defect','Defect')
    x3_f,y3_f = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/fake_images/grinding_mark','Rail with Grinding Mark')
    x4_f,y4_f = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/fake_images/shelling','Shelling')
    x5_f,y5_f = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/fake_images/squat','Squat')
    score = 0
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    score5 = 0
    for threshold in range(1,2):
        x1_f_train, x1_f_test, y1_f_train, y1_f_test = train_test_split(x1_f, y1_f, test_size=0.7, random_state=20)
        x2_f_train, x2_f_test, y2_f_train, y2_f_test = train_test_split(x2_f, y2_f, test_size=0.7, random_state=20)
        x3_f_train, x3_f_test, y3_f_train, y3_f_test = train_test_split(x3_f, y3_f, test_size=0.7, random_state=20)
        x4_f_train, x4_f_test, y4_f_train, y4_f_test = train_test_split(x4_f, y4_f, test_size=0.7, random_state=20)
        x5_f_train, x5_f_test, y5_f_train, y5_f_test = train_test_split(x5_f, y5_f, test_size=0.7, random_state=20)
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
        x1_f_train,y1_f_train = augmentation(x1_f_train,y1_f_train)
        x2_f_train,y2_f_train = augmentation(x2_f_train,y2_f_train)
        x3_f_train,y3_f_train = augmentation(x3_f_train,y3_f_train)
        x4_f_train,y4_f_train = augmentation(x4_f_train,y4_f_train)
        x5_f_train,y5_f_train = augmentation(x5_f_train,y5_f_train)
        x_train = np.concatenate((x1_f_train,x2_f_train,x3_f_train,x4_f_train,x5_f_train))
        x_test = np.concatenate((x1_f_test,x2_f_test,x3_f_test,x4_f_test,x5_f_test))
        y_train = np.concatenate((y1_f_train,y2_f_train,y3_f_train,y4_f_train,y5_f_train))
        y_test = np.concatenate((y1_f_test,y2_f_test,y3_f_test,y4_f_test,y5_f_test))
        y_train = np_utils.to_categorical(y_train,num_classes=5)
        y_test = np_utils.to_categorical(y_test,num_classes=5)
        model = load_model(filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/resnet152_02-26_all',custom_objects={'Scale':Scale})
        time_start=time.time()
        pre = model.predict(x_test)
        pre = np.argmax(pre,axis=1)
        y_test = np.argmax(y_test,axis=1)
        time_end=time.time()
        print('totally cost',(time_end-time_start)/len(y_test))
        score+=accuracy_score(y_test,pre)
        predictions_valid = model.predict(x1_f_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score1+=accuracy_score(y1_f_test, prediction)
        predictions_valid = model.predict(x2_f_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score2+=accuracy_score(y2_f_test, prediction)
        predictions_valid = model.predict(x3_f_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score3+=accuracy_score(y3_f_test, prediction)
        predictions_valid = model.predict(x4_f_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score4+=accuracy_score(y4_f_test, prediction)
        predictions_valid = model.predict(x5_f_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score5+=accuracy_score(y5_f_test, prediction)       
        print(score,score1,score2,score3,score4,score5)
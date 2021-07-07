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
    num_classes = 6
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
            if img.shape[1]<299:
                img = cv2.resize(cv2.copyMakeBorder(img,0,0,int((299-img.shape[1])/2),int((299-img.shape[1])/2),cv2.BORDER_CONSTANT,value=255),(299,299))
            else: img = cv2.resize(img,(299,299))
            data.append(img) 
            y = np.append(y,label_dict[label_name])
        return np.array(data),y
    kf = KFold(n_splits=4)
    x1_,y1_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/corrugation_new','Corrugation')
    x1,y1 = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Corrugation','Corrugation')
    x2,y2 = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Defect','Defect')
    x2_,y2_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/defect_new_','Defect')
    x3,y3 = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Rail_with_Grinding_Mark','Rail with Grinding Mark')
    x4,y4 = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Shelling','Shelling')
    x4_,y4_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/shelling_new','Shelling')
    x5,y5 = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Squat','Squat')
    x5_,y5_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/squat_new','Squat')
    x6_,y6_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/extraction_wr','normal')
    score = 0
    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    score5 = 0
    score6 = 0
    score1_ = 0
    score4_ = 0
    score5_ = 0
    for threshold in range(1,2):
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=18)
        x1_train_, x1_test_, y1_train_, y1_test_ = train_test_split(x1_, y1_, test_size=0.3, random_state=18)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=18)
        x2_train_, x2_test_, y2_train_, y2_test_ = train_test_split(x2_, y2_, test_size=0.3, random_state=18)
        x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=18)
        x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.3, random_state=18)
        x4_train_, x4_test_, y4_train_, y4_test_ = train_test_split(x4_, y4_, test_size=0.3, random_state=18)
        x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.3, random_state=18)
        x5_train_, x5_test_, y5_train_, y5_test_ = train_test_split(x5_, y5_, test_size=0.3, random_state=18)
        #x6_train, x6_test, y6_train, y6_test = train_test_split(x6, y6, test_size=0.3, random_state=20)
        x6_train_, x6_test_, y6_train_, y6_test_ = train_test_split(x6_, y6_, test_size=0.3, random_state=18)
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
        x1_train,y1_train = augmentation(x1_train,y1_train)
        x4_train,y4_train = augmentation(x4_train,y4_train)
        x5_train,y5_train = augmentation(x5_train,y5_train)      
        x_train = np.concatenate((x1_train,x2_train,x2_train_,x3_train,x4_train,x5_train,x6_train_,x1_train_,x4_train_,x5_train_))
        x_test = np.concatenate((x1_test,x2_test,x2_test_,x3_test,x4_test,x5_test,x6_test_,x1_test_,x4_test_,x5_test_))
        y_train = np.concatenate((y1_train,y2_train,y2_train_,y3_train,y4_train,y5_train,y6_train_,y1_train_,y4_train_,y5_train_))
        y_test = np.concatenate((y1_test,y2_test,y2_test_,y3_test,y4_test,y5_test,y6_test_,y1_test_,y4_test_,y5_test_)) 
        y_train = np_utils.to_categorical(y_train,num_classes=6)
        y_test = np_utils.to_categorical(y_test,num_classes=6)
        model = load_model(filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/inceptionv4_02-27_all',custom_objects={'Scale':Scale})
        time_start=time.time()
        pre = model.predict(x_test)
        pre = np.argmax(pre,axis=1)
        y_test = np.argmax(y_test,axis=1)
        time_end=time.time()
        print('totally cost',(time_end-time_start)/len(y_test))
        x1_test = np.concatenate((x1_test, x1_test_))
        y1_test = np.concatenate((y1_test, y1_test_))
        x4_test = np.concatenate((x4_test, x4_test_))
        y4_test = np.concatenate((y4_test, y4_test_))
        x5_test = np.concatenate((x5_test, x5_test_))
        y5_test = np.concatenate((y5_test, y5_test_))
        x2_test = np.concatenate((x2_test, x2_test_))
        y2_test = np.concatenate((y2_test, y2_test_))
        score+=accuracy_score(y_test,pre)
        predictions_valid = model.predict(x1_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score1+=accuracy_score(y1_test, prediction)
        predictions_valid = model.predict(x2_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score2+=accuracy_score(y2_test, prediction)
        predictions_valid = model.predict(x3_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score3+=accuracy_score(y3_test, prediction)
        predictions_valid = model.predict(x4_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score4+=accuracy_score(y4_test, prediction)
        predictions_valid = model.predict(x5_test)
        prediction = np.argmax(predictions_valid,axis=1)
        score5+=accuracy_score(y5_test, prediction)
        predictions_valid = model.predict(x6_test_)
        prediction = np.argmax(predictions_valid,axis=1)
        score6+=accuracy_score(y6_test_, prediction)       
        print(score,score1,score2,score3,score4,score5,score6)
        print(len(x1_test),len(x2_test),len(x3_test),len(x4_test),len(x5_test),len(x6_test_))
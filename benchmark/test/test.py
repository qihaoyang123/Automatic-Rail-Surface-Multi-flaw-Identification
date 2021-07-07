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
    num_classes = 6
    batch_size = 8
    nb_epoch = 100
    label_dict = {'Corrugation':0,'Defect':1,'Rail with Grinding Mark':2,'Shelling':3,'Squat':4}
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
    x1_test_,y1_test_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/corrugation_new_test','Corrugation')
    x1_test,y1_test = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Corrugation_test','Corrugation')
    x2_test,y2_test = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Defect_test','Defect')
    x2_test_,y2_test_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/defect_new_test','Defect')
    x3_test,y3_test = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Rail_with_Grinding_Mark_test','Rail with Grinding Mark')
    x4_test,y4_test = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Shelling_test','Shelling')
    x4_test_,y4_test_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/shelling_new_test','Shelling')
    x5_test,y5_test = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Squat_test','Squat')
    x5_test_,y5_test_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/squat_new_test','Squat')
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
        x_test = np.concatenate((x1_test,x2_test,x2_test_,x3_test,x4_test,x5_test,x1_test_,x4_test_,x5_test_))
        y_test = np.concatenate((y1_test,y2_test,y2_test_,y3_test,y4_test,y5_test,y1_test_,y4_test_,y5_test_))
        y_test = np_utils.to_categorical(y_test,num_classes=5)
        model = load_model(filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/densenet_169_03-01_all_5classes',custom_objects={'Scale':Scale})
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
        print(score,score1,score2,score3,score4,score5)

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
    file = pd.read_csv('/home/daisy001/qihaoyang/track_model/Data-defect/analysis_validation_select_checked.csv')
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
    x1_test,y1_test = read_image('/home/daisy001/qihaoyang/track_model/Data-defect/others_test','others')
    x2_test,y2_test = read_image('/home/daisy001/qihaoyang/track_model/Data-defect/normal_test','normal')
    kf = KFold(n_splits=4)
    for threshold in range(1,2):
        x_test = np.concatenate((x1_test,x2_test))
        y_test = np.concatenate((y1_test,y2_test))
        y_test = np_utils.to_categorical(y_test,num_classes=2)
        model = load_model(filepath = '/home/daisy001/qihaoyang/track_model/model/densenet_169_06-29_all_twoclasses',custom_objects={'Scale':Scale})
        time_start=time.time()
        pre = model.predict(x_test)
        time_end=time.time()
        print('totally cost',(time_end-time_start)/len(y_test))
        pre = np.argmax(pre,axis=1)
        y_test = np.argmax(y_test,axis=1)
        score=accuracy_score(y_test,pre)
        print(score)
# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import cv2
from keras.models import load_model
from custom_layers.scale_layer import Scale
import tensorflow as tf
from keras.models import Model
import numpy as np
from skimage.feature import local_binary_pattern,hog
from sklearn.preprocessing import scale
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import os
METHOD = 'uniform'
radius = 7
n_points = 8 * radius
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def class_weighted_crossentropy(target, output):
    axis = -1
    output /= tf.reduce_sum(output, axis, True)
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.225, 0.225, 0.1, 0.225, 0.225]
    return -tf.reduce_sum(target * weights * tf.log(output))

if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    label_dict = {'Corrugation':0,'Defect':1,'Rail with Grinding Mark':2,'Shelling':3,'Squat':4}
    feature = np.load('x_test1.npy')
    filepath = r'C:\Users\sdscit\Desktop\cnn_finetune-master\model\densenet_169_01-20_1_add_+'
    model2 = load_model(filepath,custom_objects={'class_weighted_crossentropy':class_weighted_crossentropy})
    predict = model2.predict(feature)
    predict = np.argmax(predict,axis=1)
    print(predict)
        
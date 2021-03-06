# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import os
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import time
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.metrics import log_loss, accuracy_score

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False):
    """
    Utility function to apply conv + BN. 
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      border_mode=border_mode,
                      bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def block_inception_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2,2), border_mode='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def block_reduction_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2,2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    x = merge([branch_0, branch_1, branch_2], mode='concat', concat_axis=channel_axis)
    return x


def block_inception_c(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = merge([branch_10, branch_11], mode='concat', concat_axis=channel_axis)


    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = merge([branch_20, branch_21], mode='concat', concat_axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = merge([branch_0, branch_1, branch_2, branch_3], mode='concat', concat_axis=channel_axis)
    return x


def inception_v4_base(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2,2), border_mode='valid')
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2,2), border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid')

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2,2), border_mode='valid')
    branch_1 = MaxPooling2D((3,3), strides=(2,2), border_mode='valid')(net)

    net = merge([branch_0, branch_1], mode='concat', concat_axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
      net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
      net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
      net = block_inception_c(net)

    return net


def inception_v4_model(img_rows, img_cols, color_type=1, num_classes=None, dropout_keep_prob=0.2):
    '''
    ImageNet Pretrained Weights 
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    net = inception_v4_base(inputs)


    # Final pooling and prediction

    # 8 x 8 x 1536
    net_old = AveragePooling2D((8,8), border_mode='valid')(net)

    # 1 x 1 x 1536
    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    # 1536
    predictions = Dense(output_dim=1001, activation='softmax')(net_old)

    model = Model(inputs, predictions, name='inception_v4')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/inception-v4_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8,8), border_mode='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(output_dim=num_classes, activation='softmax')(net_ft)

    model = Model(inputs, predictions_ft, name='inception_v4')

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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

    x1_,y1_ = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/corrugation_new','Corrugation')
    x1,y1 = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/Corrugation','Corrugation')
    x2,y2 = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/Defect','Defect')
    x2_,y2_ = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/defect_new_','Defect')
    x3,y3 = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/Rail_with_Grinding_Mark','Rail with Grinding Mark')
    x4,y4 = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/Shelling','Shelling')
    x4_,y4_ = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/shelling_new','Shelling')
    x5,y5 = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/Squat','Squat')
    x5_,y5_ = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/squat_new','Squat')
    x6_,y6_ = read_image('/home/daisy001/mountdir/qihaoyang/track_model/Data-defect/extraction_wr','normal')
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
        filepath = '/home/daisy001/mountdir/qihaoyang/track_model/model/inceptionv4_'+time.strftime("%m-%d",time.localtime())+'_all'
        #filepath = r'C:\Users\sdscit\Desktop\cnn_finetune-master\model\densenet_169_12-02_'+str(threshold)

        # Load our model
        model = inception_v4_model(img_rows=img_rows, img_cols=img_cols, num_classes=num_classes)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
        callbacks_list = [checkpoint]

        # Start Fine-tuning
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  shuffle=True,
                  verbose=2,
                  validation_data=(x_test, y_test),
                  callbacks = callbacks_list
                  )
        K.clear_session()
        tf.reset_default_graph()
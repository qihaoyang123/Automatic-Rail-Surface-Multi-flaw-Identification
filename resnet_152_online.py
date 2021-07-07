# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from sklearn.metrics import log_loss
from custom_layers.scale_layer import Scale
from sklearn.metrics import log_loss, accuracy_score, recall_score
from keras.models import load_model
import numpy as np
import os
import cv2
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def resnet152_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfZHhUT3lWVWxRN28/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1,8):
      x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1,36):
      x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/resnet152_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'model/resnet152_02-27_all'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
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
            if img.shape[1]<224:
                img = cv2.resize(cv2.copyMakeBorder(img,0,0,int((224-img.shape[1])/2),int((224-img.shape[1])/2),cv2.BORDER_CONSTANT,value=255),(224,224))
            else: img = cv2.resize(img,(224,224))
            data.append(img) 
            y = np.append(y,label_dict[label_name])
        return np.array(data),y

    x2_,y2_ = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/defect__','Defect')
    x1_o,y1_o = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Online Data/corrugation_new_online','Corrugation')
    x5_o,y5_o = read_image('/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/Data-defect/Online Data/squat_new_online','Squat')
    for threshold in range(1,2):
        x2_train_, x2_test_, y2_train_, y2_test_ = train_test_split(x2_, y2_, test_size=0.9, random_state=18)
        x1_o_train, x1_o_test, y1_o_train, y1_o_test = train_test_split(x1_o, y1_o, test_size=0.9, random_state=18)
        x5_o_train, x5_o_test, y5_o_train, y5_o_test = train_test_split(x5_o, y5_o, test_size=0.9, random_state=18)
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
        y_train = np.concatenate((y2_train_,y1_o_train,y5_o_train))
        y_test = np.concatenate((y2_test_,y1_o_test,y5_o_test))
        y_train = np_utils.to_categorical(y_train,num_classes=3)
        y_test = np_utils.to_categorical(y_test,num_classes=3)
        filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/resnet152_'+time.strftime("%m-%d",time.localtime())+'_all_online'

        # Load our model
        model = resnet152_model(img_rows, img_cols, channel, num_classes)
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
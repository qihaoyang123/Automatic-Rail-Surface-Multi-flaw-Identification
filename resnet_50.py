# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

from keras.optimizers import SGD, Adam
from keras.layers import Input, merge, ZeroPadding2D,Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
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
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    ImageNet Pretrained Weights 
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data 
    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
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
            if img.shape[1]<224:
                img = cv2.resize(cv2.copyMakeBorder(img,0,0,int((224-img.shape[1])/2),int((224-img.shape[1])/2),cv2.BORDER_CONSTANT,value=255),(224,224))
            else: img = cv2.resize(img,(224,224))
            data.append(img) 
            y = np.append(y,label_dict[label_name])
        return np.array(data),y

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
        filepath = '/home/daisy001/mountdir/qihaoyang/cnn_finetune-master_/model/resnet50_'+time.strftime("%m-%d",time.localtime())+'_all'

        # Load our model
        model = resnet50_model(img_rows, img_cols, channel, num_classes)
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
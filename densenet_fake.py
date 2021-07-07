# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

from keras.optimizers import SGD, Adam, Nadam
from keras.layers import Input, merge, ZeroPadding2D, Embedding, Lambda
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
from sklearn.preprocessing import scale
from keras.callbacks import ModelCheckpoint
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import tensorflow as tf
from skimage.feature import local_binary_pattern,hog
os.environ["CUDA_VISIBLE_DEVICES"]="1"
METHOD = 'uniform'
radius = 7
n_points = 8 * radius

def densenet169_model(img_rows, img_cols, weights_path, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,32,32] # For DenseNet-169

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(2, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    model.load_weights(weights_path, by_name=True)
    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)
    '''
    for layer in model.layers[:-6]:
        layer.trainable = False
    '''
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = Nadam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=1e-6)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model                 


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def mycrossentropy(target, output, e=0.1):
    loss1 = -tf.reduce_sum(target * tf.log(output))
    loss2 = -tf.reduce_sum(K.ones_like(output)/num_classes*tf.log(output))
    return (1-e)*loss1 + e*loss2

def class_weighted_crossentropy(target, output):
    axis = -1
    output /= tf.reduce_sum(output, axis, True)
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.4, 0.4, 0.2, 0.2, 0.2]
    return -tf.reduce_sum(target * weights * tf.log(output))

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

#%%
if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 5
    batch_size = 8
    nb_epoch = 100
    file = pd.read_csv('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/analysis_validation_select_checked.csv')
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
    x1_,y1_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/corrugation_new','Corrugation')
    x1,y1 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Corrugation','Corrugation')
    x2,y2 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Defect','Defect')
    x3,y3 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Rail_with_Grinding_Mark','Rail with Grinding Mark')
    x4,y4 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Shelling','Shelling')
    x4_,y4_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/shelling_new','Shelling')
    x5,y5 = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/Squat','Squat')
    x5_,y5_ = read_image('/home/daisy001/qihaoyang/cnn_finetune-master_/Data-defect/squat_new','Squat')
    kf = KFold(n_splits=4)
    def glcm(arr, d_x, d_y, gray_level=16): #(1,0) horizontal direction （0，1）vertical direction （1，1）45 degree direction （-1，1）135 degree direction
        '''return glcm matrix'''
        max_gray = arr.max()
        height, width = arr.shape
        arr = arr.astype(np.float64)
        arr = arr * (gray_level - 1) // max_gray
        ret = np.zeros([gray_level, gray_level])
        for j in range(height -  abs(d_y)):
            for i in range(width - abs(d_x)):
                rows = arr[j][i].astype(int)
                cols = arr[j + d_y][i + d_x].astype(int)
                ret[rows][cols] += 1
        if d_x >= d_y:
            ret = ret / float(height * (width - 1))
        else:
            ret = ret / float((height - 1) * (width - 1))
        return ret
        
    for threshold in range(1,2):
        x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=20)
        x1_train_, x1_test_, y1_train_, y1_test_ = train_test_split(x1_, y1_, test_size=0.3, random_state=20)
        x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=20)
        x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=20)
        x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.3, random_state=20)
        x4_train_, x4_test_, y4_train_, y4_test_ = train_test_split(x4_, y4_, test_size=0.3, random_state=20)
        x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.3, random_state=20)
        x5_train_, x5_test_, y5_train_, y5_test_ = train_test_split(x5_, y5_, test_size=0.3, random_state=20)
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
        x_train = np.concatenate((x1_train,x2_train,x3_train,x4_train,x5_train,x1_train_,x4_train_,x5_train_))
        x_test = np.concatenate((x1_test,x2_test,x3_test,x4_test,x5_test,x1_test_,x4_test_,x5_test_))
        y_train = np.concatenate((y1_train,y2_train,y3_train,y4_train,y5_train,y1_train_,y4_train_,y5_train_))
        y_test = np.concatenate((y1_test,y2_test,y3_test,y4_test,y5_test,y1_test_,y4_test_,y5_test_))        
        y_train = np_utils.to_categorical(y_train,num_classes=5)
        y_test = np_utils.to_categorical(y_test,num_classes=5)
        filepath = '/home/daisy001/qihaoyang/cnn_finetune-master_/model/densenet_169_'+time.strftime("%m-%d",time.localtime())+'_new_5classes_weighted'
        # Load our model
        model = densenet169_model(img_rows=img_rows, img_cols=img_cols, weights_path='model/densenet_169_08-06_new_twoclasses', color_type=channel, num_classes=num_classes)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
        callbacks_list = [checkpoint]
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
        
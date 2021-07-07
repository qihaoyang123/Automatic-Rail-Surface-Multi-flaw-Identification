# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import math
import torch.nn as nn
import math
from collections import OrderedDict
import torch
from torchvision import transforms
from torch.autograd import Variable
import os
import cv2
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from utee import misc
from collections import OrderedDict
import time

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes

        self.group1 = nn.Sequential(
            OrderedDict([
                ('squeeze', nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)),
                ('squeeze_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group2 = nn.Sequential(
            OrderedDict([
                ('expand1x1', nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)),
                ('expand1x1_activation', nn.ReLU(inplace=True))
            ])
        )

        self.group3 = nn.Sequential(
            OrderedDict([
                ('expand3x3', nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)),
                ('expand3x3_activation', nn.ReLU(inplace=True))
            ])
        )

    def forward(self, x):
        x = self.group1(x)
        return torch.cat([self.group2(x),self.group3(x)], 1)
        
class SqueezeNet_re(nn.Module):

    def __init__(self, version=1.0, num_classes=6):
        super(SqueezeNet_re, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
#        self.classifier = nn.Sequential(
#            nn.Dropout(p=0.5),
#            final_conv,
#            nn.ReLU(inplace=True),
#            nn.AvgPool2d(13)
#        )
        self.classifier1 = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
        ) 
        final_conv2 = nn.Conv2d(512, num_classes, kernel_size=3,padding=1)
        self.classifier2 = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv2,
            nn.ReLU(inplace=True),
        )   
        self.classifier3 = nn.Sequential(nn.AvgPool2d(13),nn.Conv2d(12, num_classes, kernel_size=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                gain = 2.0
                if m is final_conv:
                    m.weight.data.normal_(0, 0.01)
                else:
                    fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    u = math.sqrt(3.0 * gain / fan_in)
                    m.weight.data.uniform_(-u, u)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        x = torch.cat([x1,x2],dim=1)
        x = self.classifier3(x)
        return x.view(x.size(0), self.num_classes)

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
    kf = KFold(n_splits=4)
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
        
    train_transforms = transforms.Compose([
                                       transforms.ToTensor(), # 对图像进行张量化，以便神经网络处理
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transforms = transforms.Compose([
                                       transforms.ToTensor(), # 对图像进行张量化，以便神经网络处理
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    x_train_ = []
    for i in x_train:
        x_train_single = train_transforms(i)
        x_train_.append(x_train_single)
    x_test_ = []
    for i in x_test:
        x_test_single = test_transforms(i)
        x_test_.append(x_test_single)
    model = torch.load('model/squeezenet.pkl')
    base_params = list(map(id, model.features[0:5].parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    lr = 0.0001
    params = [{'params': logits_params},
          {'params': model.features[0].parameters(), 'lr': lr * 9},
          {'params': model.features[1].parameters(), 'lr': lr * 9},
          {'params': model.features[2].parameters(), 'lr': lr * 9},
          {'params': model.features[3].parameters(), 'lr': lr * 9},
          {'params': model.features[4].parameters(), 'lr': lr * 9}]
    optimizer = torch.optim.SGD(params, lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()
    model = model.cuda()
    loss_func = loss_func.cuda()
    def one_hot(x, class_count):
        return torch.eye(class_count)[x,:]
    for epoch in range(100):
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for i in range(0,len(x_train_)):
            batch_x = x_train_[i]
            batch_y = y_train[i]
            batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x)
            loss = loss_func(out, batch_y.long())
            train_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y.long()).sum()
            train_acc += train_correct.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(x_train_)), train_acc / (len(x_train_))))
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for j in range(0,len(x_test_)):
            batch_x = x_test_[j]
            batch_y = y_test[j]
            batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x)
            loss = loss_func(out, batch_y.long())
            eval_loss += loss.data.item()
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y.long()).sum()
            eval_acc += num_correct.data.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(x_test_)), eval_acc / (len(x_test_))))
        
    def evalute(x_test, y_test, model):
        eval_acc = 0.
        model.eval()
        for j in range(0,len(x_test)):
            batch_x = x_test[j]
            batch_y = y_test[j]
            batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x)
            pred = torch.max(out, 1)[1]
            num_correct = (pred == batch_y.long()).sum()
            eval_acc += num_correct.data.item()
        return eval_acc/len(x_test)
        
    def test_transformer(x_test, test_transforms):
        x_test_ = []
        for i in x_test:
            x_test_single = test_transforms(i)
            x_test_.append(x_test_single)
        return x_test_
    
    x1_test = test_transformer(x2_test_,test_transforms)
    x2_test = test_transformer(x1_o_test,test_transforms)
    x3_test = test_transformer(x5_o_test,test_transforms)
    time_start=time.time()
    print(evalute(x_test_,y_test,model))
    time_end=time.time()
    print('totally cost',(time_end-time_start)/len(y_test))
    print(evalute(x1_test,y2_test_,model), evalute(x2_test,y1_o_test,model), evalute(x3_test,y5_o_test,model))
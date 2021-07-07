# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import math
import torch.nn as nn
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


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


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


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

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
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
    
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

def squeezenet1_0(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """
    model = SqueezeNet(version=1.0, **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['squeezenet1_0'], model_root)
    return model


def squeezenet1_1(pretrained=False, model_root=None, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['squeezenet1_1'], model_root)
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
    # net = squeezenet1_0(pretrained=True)
    # model = SqueezeNet_re()
    # state_dict = net.state_dict()
    # model_dict = model.state_dict()
    # new_state_dict = OrderedDict()
    # new_state_dict = {k:v for k,v in state_dict.items() if k in model_dict}
    # model.load_state_dict(model_dict)
    # base_params = list(map(id, model.features[0:5].parameters()))
    # logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    # lr = 0.0005
    # params = [{'params': logits_params},
          # {'params': model.features[0].parameters(), 'lr': lr * 9},
          # {'params': model.features[1].parameters(), 'lr': lr * 9},
          # {'params': model.features[2].parameters(), 'lr': lr * 9},
          # {'params': model.features[3].parameters(), 'lr': lr * 9},
          # {'params': model.features[4].parameters(), 'lr': lr * 9}]
    # optimizer = torch.optim.SGD(params, lr=lr)
    # loss_func = torch.nn.CrossEntropyLoss()
    # model = model.cuda()
    # loss_func = loss_func.cuda()
    # def one_hot(x, class_count):
        # return torch.eye(class_count)[x,:]
    # for epoch in range(100):
        # print('epoch {}'.format(epoch + 1))
        # # training-----------------------------
        # train_loss = 0.
        # train_acc = 0.
        # for i in range(0,len(x_train_)):
            # batch_x = x_train_[i]
            # batch_y = y_train[i]
            # batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
            # batch_x = batch_x.cuda()
            # batch_y = batch_y.cuda()
            # out = model(batch_x)
            # loss = loss_func(out, batch_y.long())
            # train_loss += loss.data.item()
            # pred = torch.max(out, 1)[1]
            # train_correct = (pred == batch_y.long()).sum()
            # train_acc += train_correct.data.item()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        # print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(x_train_)), train_acc / (len(x_train_))))
        # model.eval()
        # eval_loss = 0.
        # eval_acc = 0.
        # for j in range(0,len(x_test_)):
            # batch_x = x_test_[j]
            # batch_y = y_test[j]
            # batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
            # batch_x = batch_x.cuda()
            # batch_y = batch_y.cuda()
            # out = model(batch_x)
            # loss = loss_func(out, batch_y.long())
            # eval_loss += loss.data.item()
            # pred = torch.max(out, 1)[1]
            # num_correct = (pred == batch_y.long()).sum()
            # eval_acc += num_correct.data.item()
        # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(x_test_)), eval_acc / (len(x_test_))))
    # torch.save(model, 'model/squeezenet.pkl')  
    model = torch.load('model/squeezenet.pkl')
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
        
    x1_test = np.concatenate((x1_test, x1_test_))
    y1_test = np.concatenate((y1_test, y1_test_))
    x4_test = np.concatenate((x4_test, x4_test_))
    y4_test = np.concatenate((y4_test, y4_test_))
    x5_test = np.concatenate((x5_test, x5_test_))
    y5_test = np.concatenate((y5_test, y5_test_))
    x2_test = np.concatenate((x2_test, x2_test_))
    y2_test = np.concatenate((y2_test, y2_test_))
    x1_test = test_transformer(x1_test,test_transforms)
    x2_test = test_transformer(x2_test,test_transforms)
    x3_test = test_transformer(x3_test,test_transforms)
    x4_test = test_transformer(x4_test,test_transforms)
    x5_test = test_transformer(x5_test,test_transforms)
    x6_test = test_transformer(x6_test_,test_transforms)
    print(evalute(x_test_,y_test,model),evalute(x1_test,y1_test,model), evalute(x2_test,y2_test,model), evalute(x3_test,y3_test,model), evalute(x4_test,y4_test,model), evalute(x5_test,y5_test,model), evalute(x6_test,y6_test_,model))
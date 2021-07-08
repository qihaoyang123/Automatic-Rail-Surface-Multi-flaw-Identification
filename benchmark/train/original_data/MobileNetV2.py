# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

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
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model

class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      #取掉model的后两层
      width_mult=1.
      block = InvertedResidual
      input_channel = 32
      self.features = [conv_bn(3, input_channel, 2)]
      interverted_residual_setting = [
              # t, c, n, s
              [1, 16, 1, 1],
              [6, 24, 2, 2],
              [6, 32, 3, 2],
              [6, 32, 3, 2],
              [6, 72, 2, 2],
              [6, 432, 2, 1],
              ]
      for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel  
      self.features = nn.Sequential(*self.features)
      self.Linear_layer1 = nn.Linear(432, 108)
      self.Linear_layer2 = nn.Linear(108, 6)
      self._initialize_weights()

  def forward(self, x):
      x = self.features(x)
      x = x.mean(3).mean(2)
      x = nn.Dropout(0.5)(x)
      x = self.Linear_layer1(x)
      x = nn.Dropout(0.5)(x)
      x = self.Linear_layer2(x)
      return x
  
  def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
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
    net = mobilenet_v2(True)
    model = Net()
    state_dict = net.state_dict()
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict = {k:v for k,v in state_dict.items() if k in model_dict}
    model.load_state_dict(model_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
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
        torch.save(model, 'model/mobilenet.pkl')
#    model = torch.load('model/mobilenet.pkl')
#    def evalute(x_test, y_test, model):
#        eval_acc = 0.
#        model.eval()
#        for j in range(0,len(x_test)):
#            batch_x = x_test[j]
#            batch_y = y_test[j]
#            batch_x, batch_y = Variable(torch.unsqueeze(batch_x, dim=0).float(), requires_grad=False), Variable(torch.Tensor([batch_y]))
#            batch_x = batch_x.cuda()
#            batch_y = batch_y.cuda()
#            out = model(batch_x)
#            pred = torch.max(out, 1)[1]
#            num_correct = (pred == batch_y.long()).sum()
#            eval_acc += num_correct.data.item()
#        return eval_acc/len(x_test)
#        
#    def test_transformer(x_test, test_transforms):
#        x_test_ = []
#        for i in x_test:
#            x_test_single = test_transforms(i)
#            x_test_.append(x_test_single)
#        return x_test_ 
#        
#    x1_test = np.concatenate((x1_test, x1_test_))
#    y1_test = np.concatenate((y1_test, y1_test_))
#    x4_test = np.concatenate((x4_test, x4_test_))
#    y4_test = np.concatenate((y4_test, y4_test_))
#    x5_test = np.concatenate((x5_test, x5_test_))
#    y5_test = np.concatenate((y5_test, y5_test_))
#    x2_test = np.concatenate((x2_test, x2_test_))
#    y2_test = np.concatenate((y2_test, y2_test_))
#    x1_test = test_transformer(x1_test,test_transforms)
#    x2_test = test_transformer(x2_test,test_transforms)
#    x3_test = test_transformer(x3_test,test_transforms)
#    x4_test = test_transformer(x4_test,test_transforms)
#    x5_test = test_transformer(x5_test,test_transforms)
#    x6_test = test_transformer(x6_test_,test_transforms)
#    print(evalute(x_test_,y_test,model),evalute(x1_test,y1_test,model), evalute(x2_test,y2_test,model), evalute(x3_test,y3_test,model), evalute(x4_test,y4_test,model), evalute(x5_test,y5_test,model), evalute(x6_test,y6_test_,model))

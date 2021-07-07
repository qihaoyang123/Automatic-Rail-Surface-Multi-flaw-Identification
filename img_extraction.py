# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:54:46 2019

@author: sdscit
"""

import os
import cv2
import numpy as np
img_path1 = os.listdir(r'C:\Users\sdscit\Desktop\Data-defect\cnn_image')
img_path2 = r'C:\Users\sdscit\Desktop\training_data_all\images_preprocess'
img_path3 = r'C:\Users\sdscit\Desktop\Data-defect\others'
for path in os.listdir(img_path2):
    if path not in img_path1:
        img = cv2.imread(os.path.join(img_path2,path))
        cv2.imwrite(os.path.join(img_path3,path),img)
        
#%%
img = []
index = 0
edge = []
num_real = 0
img_name = os.listdir(img_path3)
num = len(img_name)
for i in range(0,len(img_name)):
    edge.append([])
    path = os.path.join('others',img_name[i])
    try:
        img.append(cv2.imread(path,0))
        #简单滤波
        ret1,th1 = cv2.threshold(img[index],100,255,cv2.THRESH_BINARY)
        #Otsu 滤波
        ret2,th2 = cv2.threshold(img[index],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for j in range(0,255):
            a = th2>0
            b = th1>0
            if np.sum(a[:,j])>100:
                edge[index].append(j)
            if np.sum(b[:,j])>100:
                edge[index].append(j)
        edge_min = np.min(edge[index])
        edge_max = np.max(edge[index])
        img_object = img[index][:,edge_min:edge_max+1]
        if img_object.shape[1]<150:
            cv2.imwrite(os.path.join('others_resize',img_name[i]),img_object)
            num_real+=1
        index+=1
    except:pass
    if num_real == num: break


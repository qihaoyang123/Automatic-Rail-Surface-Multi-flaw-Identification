# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import time


img = []
index = 0
edge = []
num_real = 0
img_name = os.listdir('normal_all')
num = 400
time1 = 0
for i in range(0,len(img_name)):
    edge.append([])
    path = os.path.join('normal_all',img_name[i])
    
    try:
        start = time.time()
        img.append(cv2.resize(cv2.imread(path,0),(299,299)))
        #sample filter
        ret1,th1 = cv2.threshold(img[index],60,255,cv2.THRESH_BINARY)
        #Otsu filter
        ret2,th2 = cv2.threshold(img[index],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        for j in range(0,255):
            a = th2>0
            b = th1>0
            if np.sum(a[:,j])>90:
                edge[index].append(j)
            if np.sum(b[:,j])>90:
                edge[index].append(j)
        edge_min = np.min(edge[index])
        edge_max = np.max(edge[index])
        img_object = img[index][:,edge_min:edge_max+1]
        if img_object.shape[1]<256:
            cv2.imwrite(os.path.join('extraction_wr',img_name[i]),img_object)
            end = time.time()
            print("Execution Time: ", end - start)
            time1+=(end - start)/91
            num_real+=1
        index+=1
    except:pass
    if num_real == num: break


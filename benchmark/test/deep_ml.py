# -*- coding: utf-8 -*-
"""
@author: haoyangqi
"""

from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="0"
label_dict = {'normal':0,'Corrugation':1,'Defect':2,'Rail_with_Grinding_Mark':3,'Shelling':4,'Squat':5}

svm_score = 0
svm_score1 = 0
svm_score2 = 0
svm_score3 = 0
svm_score4 = 0
svm_score5 = 0
svm_score6 = 0
xgboost_score = 0
xgboost_score1 = 0
xgboost_score2 = 0
xgboost_score3 = 0
xgboost_score4 = 0
xgboost_score5 = 0
xgboost_score6 = 0
randomforest_score = 0
randomforest_score1 = 0
randomforest_score2 = 0
randomforest_score3 = 0
randomforest_score4 = 0
randomforest_score5 = 0
randomforest_score6 = 0
knn_score = 0
logistic_score = 0
lightgbm_score = 0
lightgbm_score1 = 0
lightgbm_score2 = 0
lightgbm_score3 = 0
lightgbm_score4 = 0
lightgbm_score5 = 0
lightgbm_score6 = 0
#%%
from custom_layers.scale_layer import Scale
from keras.models import load_model
from keras.models import Model

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

x1_,y1_ = read_image('Data-defect/corrugation_new','Corrugation')
x1,y1 = read_image('Data-defect/Corrugation','Corrugation')
x2,y2 = read_image('Data-defect/Defect','Defect')
x2_,y2_ = read_image('Data-defect/defect_new_','Defect')
x3,y3 = read_image('Data-defect/Rail_with_Grinding_Mark','Rail_with_Grinding_Mark')
x4,y4 = read_image('Data-defect/Shelling','Shelling')
x4_,y4_ = read_image('Data-defect/shelling_new','Shelling')
x5,y5 = read_image('Data-defect/Squat','Squat')
x5_,y5_ = read_image('Data-defect/squat_new','Squat')
x6_,y6_ = read_image('Data-defect/extraction_wr','normal')
print(1)
filepath = 'model/densenet_169_02-27_all'
model = load_model(filepath,custom_objects={'Scale': Scale})
feature_model = Model(inputs=model.input,outputs=model.get_layer('pool5').output)
def get_feature(data,model):
    feature = []
    output = feature_model.predict(data)
    for i in range(0,len(data)):
        matrix = list(output[i].ravel())
        feature.append(matrix)
    feature = np.array(feature)
    '''
    Axx,Axy,Ayy = structure_tensor(data, mode='nearest')
    structure_tensor_data = structure_tensor_eigvals(Axx, Axy, Ayy)[1].ravel()
    feature = np.append(feature,structure_tensor_data)
    '''
    return feature
print(1)
x1_ = get_feature(x1_,feature_model)
x1 = get_feature(x1,feature_model)
x2 = get_feature(x2,feature_model)
x2_ = get_feature(x2_,feature_model)
x3 = get_feature(x3,feature_model)
x4 = get_feature(x4,feature_model)
x4_ = get_feature(x4_,feature_model)
x5 = get_feature(x5,feature_model)
x5_ = get_feature(x5_,feature_model)
x6_ = get_feature(x6_,feature_model)
#%%
print(0)
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
x_train = np.concatenate((x1_train,x2_train,x2_train_,x3_train,x4_train,x5_train,x6_train_,x1_train_,x4_train_,x5_train_))
x_test = np.concatenate((x1_test,x2_test,x2_test_,x3_test,x4_test,x5_test,x6_test_,x1_test_,x4_test_,x5_test_))
y_train = np.concatenate((y1_train,y2_train,y2_train_,y3_train,y4_train,y5_train,y6_train_,y1_train_,y4_train_,y5_train_))
y_test = np.concatenate((y1_test,y2_test,y2_test_,y3_test,y4_test,y5_test,y6_test_,y1_test_,y4_test_,y5_test_)) 
x1_test = np.concatenate((x1_test, x1_test_))
y1_test = np.concatenate((y1_test, y1_test_))
x4_test = np.concatenate((x4_test, x4_test_))
y4_test = np.concatenate((y4_test, y4_test_))
x5_test = np.concatenate((x5_test, x5_test_))
y5_test = np.concatenate((y5_test, y5_test_))
x2_test = np.concatenate((x2_test, x2_test_))
y2_test = np.concatenate((y2_test, y2_test_))
#%%
print(1)
from sklearn.metrics import log_loss
from sklearn.svm import SVC

clf = SVC(kernel='poly')
clf.fit(x_train,y_train)
svm_score += clf.score(x_test,y_test)
svm_score1 += clf.score(x1_test,y1_test)
svm_score2 += clf.score(x2_test,y2_test)
svm_score3 += clf.score(x3_test,y3_test)
svm_score4 += clf.score(x4_test,y4_test)
svm_score5 += clf.score(x5_test,y5_test)
svm_score6 += clf.score(x6_test_,y6_test_)

print(2)
import xgboost as xgb

clf = xgb.XGBClassifier()
'''
xgb_param = clf.get_xgb_params() 
extra = {'num_class': 2} 
xgb_param.update(extra) 
'''
clf.fit(x_train,y_train)
xgboost_score += clf.score(x_test,y_test)
xgboost_score1 += clf.score(x1_test,y1_test)
xgboost_score2 += clf.score(x2_test,y2_test)
xgboost_score3 += clf.score(x3_test,y3_test)
xgboost_score4 += clf.score(x4_test,y4_test)
xgboost_score5 += clf.score(x5_test,y5_test)
xgboost_score6 += clf.score(x6_test_,y6_test_)

print(3)
from sklearn.ensemble import RandomForestClassifier as rfc

clf = rfc()

clf.fit(x_train,y_train)
randomforest_score += clf.score(x_test,y_test)
randomforest_score1 += clf.score(x1_test,y1_test)
randomforest_score2 += clf.score(x2_test,y2_test)
randomforest_score3 += clf.score(x3_test,y3_test)
randomforest_score4 += clf.score(x4_test,y4_test)
randomforest_score5 += clf.score(x5_test,y5_test)
randomforest_score6 += clf.score(x6_test_,y6_test_)

print(4)
import lightgbm as lgb

clf = lgb.LGBMClassifier()

clf.fit(x_train,y_train)
lightgbm_score += clf.score(x_test,y_test)
lightgbm_score1 += clf.score(x1_test,y1_test)
lightgbm_score2 += clf.score(x2_test,y2_test)
lightgbm_score3 += clf.score(x3_test,y3_test)
lightgbm_score4 += clf.score(x4_test,y4_test)
lightgbm_score5 += clf.score(x5_test,y5_test)
lightgbm_score6 += clf.score(x6_test_,y6_test_)

#%%
print(svm_score,svm_score1,svm_score2,svm_score3,svm_score4,svm_score5,svm_score6)
print(xgboost_score,xgboost_score1,xgboost_score2,xgboost_score3,xgboost_score4,xgboost_score5,xgboost_score6)
print(randomforest_score,randomforest_score1,randomforest_score2,randomforest_score3,randomforest_score4,randomforest_score5,randomforest_score6)
print(lightgbm_score,lightgbm_score1,lightgbm_score2,lightgbm_score3,lightgbm_score4,lightgbm_score5,lightgbm_score6)
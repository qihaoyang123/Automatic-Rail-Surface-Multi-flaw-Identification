# The Automatic Rail Surface Multi-flaw Identification Based on A Deep Learning Powered Framework

This repository contains the code for DenseNet introduced in the following paper:

The Automatic Rail Surface Multi-flaw Identification Based on A Deep Learning Powered Framework

## Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
4. [Results](#results)

## Introduction
This paper develops a vision-based inspection framework for the automated identification of rail surface flaw categories.

The procedure of the model in this paper is presented in the next image:
<img src="https://github.com/qihaoyang123/Automatic-Rail-Surface-Multi-flaw-Identification/blob/main/images/procedure.jpg" width="600" height="400">

The structure of the classifier in this paper is presented in the next image:
<img src="https://github.com/qihaoyang123/Automatic-Rail-Surface-Multi-flaw-Identification/blob/main/images/structure.jpg" width="600" height="400">

## Requirements
* python 3.6.12
* keras 2.1.4
* tensorflow-gpu 1.14.0
* opencv-python
* Scikit-Image
* sklearn
* lightgbm
* xgboost
* pandas
* IPython
* pytorch 1.0.1
* torchvision 0.4.0

## Usage
1. Install Keras and required dependencies
2. Clone this repo: https://github.com/qihaoyang123/Automatic-Rail-Surface-Multi-flaw-Identification
3. Download the test set: https://drive.google.com/drive/folders/1FvviSvhgnhe2424K-sMQ7viJAhcZ6lP1?usp=sharing
   Download the weight trained by our data to test the results of our experiments: https://drive.google.com/drive/folders/1ra-qUmwvx9-7JYThDQ77zqHJLnmyEA6x?usp=sharing
   Download the initial weight to train your own data: https://drive.google.com/drive/folders/11mmlNC_PCGG5-LpD6ynYN7h_lRtUDDa2?usp=sharing
   After downloading all files required for code running, you need to change the path of these files in the code files.
4. Due to data confidentiality, we only release the test set. The files for testing are in this path: benchmark/test. The following command tests the benchmark:
```
python test.py
``` 
```
python test_benchmark.py
``` 
```
python deep_ml.py
``` 
5. The code for testing the proposed framework is in the proposed_framework/test folder. The following command tests our model:
	
```
python densenet_add_test.py
```

6. If you want to test the online dataset, you can just run the test_online.py, deep_ml_online.py, densenet_add_online_test.py, which are in the path of benchmark/test and proposed_framework/test: .

7. If you want to use your own data to train the model, you need to run track_extraction.py to extract the critical part of the image. Then you can utilize images to train our model. The training codes are in benchmark/train and proposed_framework/train folders respectively.

## Results

![image](https://github.com/qihaoyang123/Automatic-Rail-Surface-Multi-flaw-Identification/blob/main/images/result.png)

## Contact
qihaoyanggene@gmail.com

lizhuanglily@gmail.com


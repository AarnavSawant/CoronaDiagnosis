#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:00:13 2020

@author: parisawant
"""

import os
import numpy as np
import shutil

root_dir = 'dataset'
COVID_diagnosis = "/COVID-19"
Pneumonia_diagnosis = "/ViralPneumonia"
NOTHING_diagnosis = "/NORMAL"

os.makedirs(root_dir + "/train_data" + COVID_diagnosis)
os.makedirs(root_dir + "/train_data" + NOTHING_diagnosis)
os.makedirs(root_dir + "/train_data" + Pneumonia_diagnosis)
os.makedirs(root_dir + "/test_data" + COVID_diagnosis)
os.makedirs(root_dir + "/test_data" + Pneumonia_diagnosis)
os.makedirs(root_dir + "/test_data" + NOTHING_diagnosis)
os.makedirs(root_dir + "/validation_data" + COVID_diagnosis)
os.makedirs(root_dir + "/validation_data" + Pneumonia_diagnosis)
os.makedirs(root_dir + "/validation_data" + NOTHING_diagnosis)

diagnosis = [COVID_diagnosis, Pneumonia_diagnosis, NOTHING_diagnosis]
for current_diagnosis in diagnosis:
    src = root_dir + current_diagnosis
    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames) * 0.7), int(len(allFileNames) * 0.85)])
    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, "dataset/train_data" + current_diagnosis)

    for name in val_FileNames:
        shutil.copy(name, "dataset/validation_data" + current_diagnosis)

    for name in test_FileNames:
        shutil.copy(name, "dataset/test_data" + current_diagnosis)
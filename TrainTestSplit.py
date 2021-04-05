# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:13:17 2021

@author: vedhs
"""
from shutil import copyfile
import os
import random

def splitSets(fromPath,trainPath,validPath,ratio=0.75,print_status=True):
    plants=list(os.walk(fromPath))[0][1]

    for plant in plants:
        diseases=list(os.walk(os.path.join(fromPath,plant)))[0][1]
        
        for disease in diseases:
            if print_status:
                print('Splitting',plant,disease)
            images=list(os.walk(os.path.join(fromPath,os.path.join(plant,disease))))[0][2]
            random.shuffle(images)
            if not os.path.isdir(os.path.join(trainPath,os.path.join(plant,disease))):
                os.makedirs(os.path.join(trainPath,os.path.join(plant,disease)))
                os.makedirs(os.path.join(validPath,os.path.join(plant,disease)))
            
            i=0
            for image in images:
                if i/len(images)<ratio:
                    copyfile(os.path.join(os.path.join(fromPath,plant),os.path.join(disease,image)),os.path.join(os.path.join(trainPath,plant),os.path.join(disease,image)))
                else:
                    copyfile(os.path.join(os.path.join(fromPath,plant),os.path.join(disease,image)),os.path.join(os.path.join(validPath,plant),os.path.join(disease,image)))
                i+=1
            
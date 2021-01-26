# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:36:14 2021

@author: vedhs
"""

import PreProcess
from tensorflow import keras
import numpy as np

def predict_plant(image,skip_preprocess=False):
    if not skip_preprocess:
        image=PreProcess.preProcess(image)
    model = keras.models.load_model('plantDet2.h5')
    plants = ['Apple','Cherry','Corn','Grape','Peach','PepperBell','Potato','Strawberry']    
    pred = model.predict(np.array([image],dtype=np.float32))[0]
    
    if max(pred)>0.5:
        return predict_disease(image,np.argmax(pred),plants[np.argmax(pred)])
    else:
        return 'Failed to identify the plant'
    
def get_disease_model(index):
    if index==2:
        return keras.models.load_model('CornDet.h5')
    
def get_disease_names(index):
    if index==2:
        return ['Leaf spot','Common rust','Healthy','Leaf blight']
    
def predict_disease(image,index,plant_name):
    model = get_disease_model(index)
    disease_names = get_disease_names(index)
    pred = model.predict(np.array([image],dtype=np.float32))[0]
    if max(pred)>0.5:
        return plant_name+', '+disease_names[np.argmax(pred)]
    else:
        return plant_name+', '+'Failed to identify the disease'
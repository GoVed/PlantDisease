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
    model = keras.models.load_model('plantDet3.h5')
    plants = ['Apple','Cherry','Corn','Grape','Peach','PepperBell','Potato','Strawberry'] 
    
    pred = model.predict(np.array([image],dtype=np.float32))[0]
    print('\n')
    print('Plant class',pred) 
    if max(pred)>0:
        if max(pred)<50:
            print('Less confidence')
        return predict_disease(image,np.argmax(pred),plants[np.argmax(pred)])
    else:
        return 'Failed to identify the plant'
    
def get_disease_model(index):
    modelName=['AppleDet.h5','CherryDet.h5','CornDet.h5','','','','','']
    
    return keras.models.load_model(modelName[index])
    
def get_disease_names(index):
    disease_names=[['Scab','Black rot','Cedar apple rust','Healthy'],['Healthy','MilDew'],['Leaf spot','Common rust','Healthy','Leaf blight'],[],[],[],[],[]]
    return disease_names[index]
    
def predict_disease(image,index,plant_name):
    model = get_disease_model(index)
    disease_names = get_disease_names(index)
    pred = model.predict(np.array([image],dtype=np.float32))[0]
    print('Disease class',pred) 
    if max(pred)>0:
        if max(pred)<50:
            print('Less confidence')
        return 'Predicted:'+plant_name+', '+disease_names[np.argmax(pred)]
    else:
        return plant_name+', '+'Failed to identify the disease'
  
from PIL import Image
test1=np.asarray(Image.open('test1.jpg'))
test2=np.asarray(Image.open('test2.jpg'))
test3=np.asarray(Image.open('test3.jpg'))
test4=np.asarray(Image.open('test4.jpg'))
test5=np.asarray(Image.open('test5.jpg'))
print(predict_plant(test1))
print(predict_plant(test2))
print(predict_plant(test3))
print(predict_plant(test4))
print(predict_plant(test5))

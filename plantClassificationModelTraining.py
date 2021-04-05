# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:56:22 2021

@author: vedhs
"""

from PIL import Image
import glob
import numpy as np
from tensorflow import keras
import random


def get_plant_name(dir):
    return glob.glob(dir+"/*")
    

def get_all_images_as_array(dir,batch_size=100,print_status=True):
    x=[]
    y=[]
    plant_names=glob.glob(dir+"/*")
    
    
    i=0
    for plant_name in plant_names:  
        disease_names = glob.glob(plant_name+'/*')
        
        for disease_name in disease_names:
            disease_dir = disease_name
            
            image_ids = glob.glob(disease_dir+'/*')
            random.shuffle(image_ids)
            if print_status:
                print('Getting images from',disease_name)
                
            j=0
            for image_id in image_ids:
                
                x.append(np.asarray(Image.open(image_id).resize((256,256))))
                
                output = np.random.rand(len(plant_names))
                output*=5
                output[i]=random.randint(95, 100)
                
                # output= np.zeros(len(disease_names))
                # output[i]=1
                y.append(output)
                j+=1
                if j >= batch_size/len(disease_names)/len(plant_names):
                    break
        i+=1
    
    x=np.array(x,dtype=np.float32)
    y=np.array(y,dtype=np.float32)
    return x,y



def train_detection_model(plant_folder_train,plant_folder_valid,batches=10,batch_size=1000,test_ratio=0.2,epochs=100):

    #making the model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(256,256,3)))
    model.add(keras.layers.MaxPool2D((2,2),padding='same'))
    model.add(keras.layers.Conv2D(16,(3,3),padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D((2,2),padding='same'))
    model.add(keras.layers.Conv2D(8,(3,3),padding='same',activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(len(get_plant_name(plant_folder_train)),activation='relu'))
    
    #setting the model
    model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.mean_absolute_error)
    
    
    
    
    #per batch training
    for i in range(batches):
        x,y = get_all_images_as_array(plant_folder_train,batch_size)
        x_test,y_test = get_all_images_as_array(plant_folder_valid,batch_size*test_ratio)
        
        model.fit(x,y,epochs=epochs,shuffle=True,validation_data=(x_test,y_test))

    return model
    
    

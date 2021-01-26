# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:05:35 2021

@author: vedhs
"""

import PreProcess
import os
import cv2

def preProcess_all(path,folder_name,type='crop',segment_mul=1):
    image_ids=os.listdir(path+'/'+folder_name)
    
    for image_id in image_ids:
        image = cv2.imread(path+'/'+folder_name+'/'+image_id)
        cv2.imwrite('processed/'+folder_name+'/'+image_id,PreProcess.preProcess(image,type,segment_mul))

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:13:36 2021

@author: vedhs
"""
import numpy as np
import cv2 as cv



def segmentImage(image,l_h=37,l_s=37,l_v=58,u_h=58,u_s=255,u_v=255):
    
    image=cv.resize(image,(256,256))
    
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    
    l_b=np.array([l_h,l_s,l_v])
    u_b=np.array([u_h,u_s,u_v])

    mask=cv.inRange(hsv,l_b,u_b)

    res=cv.bitwise_and(image,image,mask=mask)
    return res

def smoothMask(mask):
    
    #changing binary mask to image
    mask=np.array(mask,dtype=np.float32)
    mask[mask==1]=255
    
    #making kernal
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    (thresh, binRed) = cv.threshold(mask, 128, 255, cv.THRESH_BINARY)
    
    #removing openings
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    
    #removing closings
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    closing[closing==255]=1
    closing=np.array(closing,np.bool)
    
    #returning the final result
    return closing

def overlayImageOnMask(image,mask):
    mask=np.dstack((mask,mask,mask)).astype(dtype=np.uint8)
    mask[mask==1]=255
    return cv.bitwise_and(image,mask)

def segmentFF(image,threshold_r=-1,threshold_g=-1,threshold_b=-1,x=127,y=127,save_path="",save_rate=0,pixel_size=3,jump=3,auto_threshold_mul=4.2):
    
    #auto set threshold
    threshold_r = image[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size,0].mean()/auto_threshold_mul
    threshold_g = image[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size,1].mean()/auto_threshold_mul
    threshold_b = image[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size,2].mean()/auto_threshold_mul
    
    #initializing mask and result image
    mask=np.zeros((image.shape[0],image.shape[1]),dtype=np.bool)
    res=np.zeros((image.shape[0],image.shape[1],image.shape[2]),dtype=np.uint8)
    
    #Queue for pixel left
    leftCheck=[[x,y]]   
    visited=[]
    i=0
    
    #for making animation
    if save_path != "":
        vid=cv.VideoWriter(save_path,0,30,(mask.shape))
    
    #Until queue is empty
    while len(leftCheck)!=0:        
        mask[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size]=1
        curr_pixel = image[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size,:]
        res[x-pixel_size:x+pixel_size,y-pixel_size:y+pixel_size,:]=curr_pixel
        x,y=leftCheck[0]  
        
        if x<image.shape[0]-jump-pixel_size:
            r_pixel = image[x-pixel_size+jump:x+pixel_size+jump,y-pixel_size:y+pixel_size,:]
            dr=abs(r_pixel[:,:,0].mean()-curr_pixel[:,:,0].mean())
            dg=abs(r_pixel[:,:,1].mean()-curr_pixel[:,:,1].mean())
            db=abs(r_pixel[:,:,2].mean()-curr_pixel[:,:,2].mean())
            
            if dr<threshold_r and dg<threshold_g and db<threshold_b and [x+jump,y] not in visited and [x+jump,y] not in leftCheck:
                leftCheck.append([x+jump,y])   
            
        if y<image.shape[1]-jump-pixel_size:
            u_pixel = image[x-pixel_size:x+pixel_size,y-pixel_size+jump:y+pixel_size+jump,:]
            dr=abs(u_pixel[:,:,0].mean()-curr_pixel[:,:,0].mean())
            dg=abs(u_pixel[:,:,1].mean()-curr_pixel[:,:,1].mean())
            db=abs(u_pixel[:,:,2].mean()-curr_pixel[:,:,2].mean())
            
            if dr<threshold_r and dg<threshold_g and db<threshold_b and [x,y+jump] not in visited and [x,y+jump] not in leftCheck:
                leftCheck.append([x,y+jump]) 
        
        if x>pixel_size+jump:
            l_pixel = image[x-pixel_size-jump:x+pixel_size-jump,y-pixel_size:y+pixel_size,:]
            dr=abs(l_pixel[:,:,0].mean()-curr_pixel[:,:,0].mean())
            dg=abs(l_pixel[:,:,1].mean()-curr_pixel[:,:,1].mean())
            db=abs(l_pixel[:,:,2].mean()-curr_pixel[:,:,2].mean())            
            if dr<threshold_r and dg<threshold_g and db<threshold_b and [x-jump,y] not in visited and [x-jump,y] not in leftCheck:
                leftCheck.append([x-jump,y])                                
        
        if y>pixel_size+jump:
            d_pixel = image[x-pixel_size:x+pixel_size,y-pixel_size-jump:y+pixel_size-jump,:]
            dr=abs(d_pixel[:,:,0].mean()-curr_pixel[:,:,0].mean())
            dg=abs(d_pixel[:,:,1].mean()-curr_pixel[:,:,1].mean())
            db=abs(d_pixel[:,:,2].mean()-curr_pixel[:,:,2].mean())
            if dr<threshold_r and dg<threshold_g and db<threshold_b and [x,y-jump] not in visited and [x,y-jump] not in leftCheck:
                leftCheck.append([x,y-jump])   
        
        visited.append([x,y])
        
        #check adjacent
        
        
        #check adjacent values              
        # if x<image.shape[0]-1:
        #     if abs(int(image[x,y,0])-int(image[x+1,y,0]))<threshold_r and abs(int(image[x,y,1])-int(image[x+1,y,1]))<threshold_g and abs(int(image[x,y,2])-int(image[x+1,y,2]))<threshold_b:
        #         if mask[x+1,y]==0 and [x+1,y] not in leftCheck:
        #             leftCheck.append([x+1,y])                    
        # if y<image.shape[1]-1:
        #     if abs(int(image[x,y,0])-int(image[x,y+1,0]))<threshold_r and abs(int(image[x,y,1])-int(image[x,y+1,1]))<threshold_g and abs(int(image[x,y,2])-int(image[x,y+1,2]))<threshold_b:
        #         if mask[x,y+1]==0 and [x,y+1] not in leftCheck:
        #             leftCheck.append([x,y+1])
        # if x>0:
        #     if abs(int(image[x,y,0])-int(image[x-1,y,0]))<threshold_r and abs(int(image[x,y,1])-int(image[x-1,y,1]))<threshold_g and abs(int(image[x,y,2])-int(image[x-1,y,2]))<threshold_b:
        #         if mask[x-1,y]==0 and [x-1,y] not in leftCheck:
        #             leftCheck.append([x-1,y])
        # if y>0:
        #     if abs(int(image[x,y,0])-int(image[x,y-1,0]))<threshold_r and abs(int(image[x,y,1])-int(image[x,y-1,1]))<threshold_g and abs(int(image[x,y,2])-int(image[x,y-1,2]))<threshold_b:
        #         if mask[x,y-1]==0 and [x,y-1] not in leftCheck:
        #             leftCheck.append([x,y-1])
                    
        #remove from the queue
        leftCheck.pop(0)
        if save_path != "":
            if i%save_rate == 0:
                # cv.imwrite(str(save_path)+str(i/save_rate).zfill(8)+'.png', res)
                
                vid.write(res)
                
                print(len(visited))
        i+=1
    if save_path != "":
        vid.release()
    return res,mask

inp = cv.resize(cv.imread('raw/healthy_leaf/i213.JPG'),(256,256))
out,mask=segmentFF(inp,save_path='tempAnim/anim.avi',save_rate=75)
cv.imwrite('tempAnim/final.png',overlayImageOnMask(inp,smoothMask(mask)))

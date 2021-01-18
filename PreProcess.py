import ImageCorrection
import ImageSegmentation
import cv2
import numpy as np

def preProcess(img,type='crop',segment_mul=1):
    
    img = ImageCorrection.automatic_brightness_and_contrast(img)
    
    if type=='crop':
        img = crop_square_center(img)
    if type=='fill':
        img = fill_square_center(img)
    img = cv2.resize(img,(256,256))
    
    _,mask = ImageSegmentation.segmentFF(img,auto_threshold_mul=segment_mul,pixel_size=6)
    return ImageSegmentation.overlayImageOnMask(img,ImageSegmentation.smoothMask(mask))
    
def crop_square_center(img):
    if img.shape[0]>img.shape[1]:
        img=img[int(img.shape[0]/2)-int(img.shape[1]/2):int(img.shape[0]/2)+int(img.shape[1]/2),:,:]
    if img.shape[1]>img.shape[0]:
        img=img[:,int(img.shape[1]/2)-int(img.shape[0]/2):int(img.shape[1]/2)+int(img.shape[0]/2),:]
    return img

def fill_square_center(img):
    temp = np.zeros((max(img.shape[0],img.shape[1]),max(img.shape[0],img.shape[1]),img.shape[2]),dtype=np.uint8)
    
    if img.shape[0]>img.shape[1]:
        temp[:,int(img.shape[0]/2)-int(img.shape[1]/2):int(img.shape[0]/2)+int(img.shape[1]/2),:]=img
    if img.shape[1]>img.shape[0]:
        temp[int(img.shape[1]/2)-int(img.shape[0]/2):int(img.shape[1]/2)+int(img.shape[0]/2),:,:]=img
    return temp
            
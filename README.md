# PlantDisease (Indus Hackathon 2021 project)

## File info:

### trackBar.py
- Use to remove background from original images
- Methods
  - Tracking
- Imported
  - numpy
  - CV2

### ImageSegmentation.py

- Use for image segmentation
- Methods
  - segmentImage(image,l_h=37,l_s=37,l_v=58,u_h=58,u_s=255,u_v=255) [Simple hsv based segmentation]
  - smoothMask(mask) [Smooth out mask border and fill the closings]
  - overlayImageOnMask(image,mask) [Put image on mask]
  - segmentFF(image,threshold_r=-1,threshold_g=-1,threshold_b=-1,x=127,y=127,save_path="",save_rate=0,pixel_size=3,jump=3,auto_threshold_mul=4.2) [Segementation using BFS/Flood fill]
- Imports
  - CV2
  - numpy

### ImageCorrection.py

- Use for image correction (brightness and contrast)
- Methods
  - automatic_brightness_and_contrast(image, clip_hist_percent=25)
  - convertScale(img, alpha, beta)
- Imports
  - CV2
  - numpy
  
### PreProcess.py
- A class which helps to PreProcess images
- Methods
  - preProcess(img,type='crop',segment_mul=1)
  - crop_square_center(img)
  - fill_square_center(img)
- Imported
  - ImageCorrection
  - ImageSegmentation
  - cv2
  - numpy
  
### Pathological innovation to support Aatma Nirbhar Bharat.pptx
- Presentation for the project

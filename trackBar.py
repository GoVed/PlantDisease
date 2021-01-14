import numpy as np
import cv2 as cv
cv.namedWindow("Tracking")
def nothing(x):
    pass

#cap=cv.VideoCapture(0)
cv.createTrackbar("LH","Tracking",0,255,nothing)
cv.createTrackbar("LS","Tracking",0,255,nothing)
cv.createTrackbar("LV","Tracking",0,255,nothing)
cv.createTrackbar("UH","Tracking",255,255,nothing)
cv.createTrackbar("US","Tracking",255,255,nothing)
cv.createTrackbar("UV","Tracking",255,255,nothing)

while(1):
    frame=cv.imread("i87.jpg")
    frame=cv.resize(frame,(256,256))
    #_,frame=cap.read()
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("LH","Tracking")
    l_s = cv.getTrackbarPos("LS", "Tracking")
    l_v = cv.getTrackbarPos("LV", "Tracking")
    u_h = cv.getTrackbarPos("UH", "Tracking")
    u_s = cv.getTrackbarPos("US", "Tracking")
    u_v = cv.getTrackbarPos("UV", "Tracking")

    l_b=np.array([l_h,l_s,l_v])
    u_b=np.array([u_h,u_s,u_v])

    mask=cv.inRange(hsv,l_b,u_b)

    res=cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow("frame",frame)
    cv.imshow("mask",mask)
    cv.imshow("res",res)
    cv.imwrite('leaf87.jpg',res)
    cv.imwrite("l87.jpg", mask)
    key=cv.waitKey(1)
    if key==27:
        break

#cap.release()
cv.destroyAllWindows()
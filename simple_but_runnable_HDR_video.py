# follow is the simple HDR video example of using web camera to 
# combine 2 frames to 1 HDR frame
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import time
np.seterr(divide='ignore', invalid='ignore')
pathHDR = "G:\\ECE516\\HDR video\\picture\\"
pathsave ="G:\\ECE516\\HDR video\\"


# config the camera
cap = cv2.VideoCapture(0)
#cap.set(3,240)
#cap.set(4,320)
#cap.set(cv2.CAP_PROP_SETTINGS,0)
cap.set(cv2.CAP_PROP_EXPOSURE,-1)
_,frame = cap.read()
_,frame = cap.read()
cv2.waitKey(10)
cv2.imshow("capture", frame)
cv2.waitKey(0)
#cv2.imwrite(pathHDR + "cs5.jpg",frame)
cap.release()
cv2.destroyAllWindows()


# show the CCRF table we are using
CCRF = np.load(pathsave + "CCRF table for HP web camera.npy")
imshow(CCRF,cmap = "Greys")
plt.show()


# show one frame
cap = cv2.VideoCapture(1)
#cap.set(cv2.CAP_PROP_SETTINGS,0)
while(cap.isOpened()):
    #cap.set(cv2.CAP_PROP_EXPOSURE,0)
    _,frame1 = cap.read()
    cv2.imshow('My Camera1xxx',frame1)
    #cv2.waitKey(15)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# simple HDR video by web camera
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_SETTINGS,0)
cap.set(3,320)
cap.set(4,240)
frame3 = np.zeros((240,320,3),dtype = np.uint8) #1,2

while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_EXPOSURE,-6)
    _,frame1 = cap.read()
    _,frame1 = cap.read()
    _,frame1 = cap.read()
    _,frame1 = cap.read()
    cap.set(cv2.CAP_PROP_EXPOSURE,-5)
    _,frame2 = cap.read()
    _,frame2 = cap.read()
    _,frame2 = cap.read()
    _,frame2 = cap.read()
    for i in range(240):
        for j in range(320):
            for k in range(3):
                frame3[i][j][k] = CCRF[frame2[i][j][k]][frame1[i][j][k]]
    cv2.imshow('My Camera1_low',frame1)
    cv2.waitKey(40)
    cv2.imshow('My Camera_high',frame2)
    cv2.waitKey(40)
    cv2.imshow('My Camera1_CCRF',frame3)
    cv2.waitKey(40)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

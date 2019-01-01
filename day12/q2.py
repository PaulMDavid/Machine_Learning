#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day11/opencv-logo.png')
img2=cv.imread('/home/ai11/Desktop/common/ML/Day11/ml.png')
print img.shape
print img2.shape
im=cv.addWeighted(img,0.7,img2,0.3,0)
cv.imshow('ds',im)
cv.waitKey(0)

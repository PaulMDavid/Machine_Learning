#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day13/butterfly.jpg')
for i in range(40):
 img=cv.pyrDown(img)
cv.imshow('adfs',img)
cv.waitKey(0)

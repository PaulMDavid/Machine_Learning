#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/apple.jpg')
img2=cv.imread('/home/ai11/Desktop/common/ML/Day10/Questions/orange.jpg')
print img.shape
img[:,256:512]= img2[:,256:512]
cv.imshow('dsp',img)
cv.waitKey(0)



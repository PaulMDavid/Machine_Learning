#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/noisy2.png')
cv.imshow('ds1',img)
img=cv.bitwise_not(img)
cv.imshow('ds',img)
cv.waitKey(0)


#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/apple.jpg')
dst=cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_CUBIC)
cv.imshow('display',dst)
cv.waitKey(0)
cv.destroyAllWindows()

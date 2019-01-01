#!/usr/bin/env python
import numpy as np 
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/apple.jpg',)
rows,cols,cl= img.shape
M=np.float32([[1,0,100],[0,1,50]])
dst=cv.warpAffine(img,M,(rows,cols))
cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAlllWindows()


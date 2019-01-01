#!/user/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/apple.jpg')
rows,cols,c=img.shape
M=cv.getRotationMatrix2D(((cols-1)/2,(rows-1)/2),45,1)
dst=cv.warpAffine(img,M,(cols,rows))
cv.imshow('Disp',dst)
cv.waitKey(0)

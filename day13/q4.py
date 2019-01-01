#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
img=cv2.imread('/home/ai11/Desktop/common/ML/Day13/building.jpg')
img=cv2.pyrDown(img)
sobelx=cv2.Sobel(img,cv2.CV_8U,1,0,ksize=1)
sobely=cv2.Sobel(img,cv2.CV_8U,0,1,ksize=1)
sobel=cv2.addWeighted(sobelx,1.0,sobely,1.0,0)
cv2.imshow('ad',sobel)
cv2.waitKey(0)

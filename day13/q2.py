#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
img=cv2.imread('/home/ai11/Desktop/common/ML/Day13/man.jpg')
img2=np.bitwise_not(img)
kernel=np.ones([2,2],np.uint8)
er=cv2.morphologyEx(img2,cv2.MORPH_OPEN,kernel)
er1=np.bitwise_not(er)
cv2.imshow('af',er1)
cv2.waitKey(0)
e=cv2.morphologyEx(img2,cv2.MORPH_CLOSE,kernel)
e1=np.bitwise_not(e)
cv2.imshow('ad',e1)
cv2.waitKey(0)

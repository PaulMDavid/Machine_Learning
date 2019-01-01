#!/usr/bin/env python
import cv2
import numpy as np
import pandas as pd
img=cv2.imread('/home/ai11/Desktop/common/ML/Day13/man.jpg')
img2=np.bitwise_not(img)
kernel=np.ones([2,2],np.uint8)
er=cv2.erode(img2,kernel,iterations=2)
mg_dilation = cv2.dilate(er, kernel, iterations=2)
er1=np.bitwise_not(er)
cv2.imshow('af',er1)
cv2.waitKey(0)

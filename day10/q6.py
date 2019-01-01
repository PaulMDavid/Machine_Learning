#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day10/images/apple.jpg')
r,c,x=img.shape
print r,c,x
x=raw_input('give add to compare')
img2=cv.imread(x)
r1,c1,x1=img2.shape
print r1,c1,x1
if(r1==r):
 if(c1==c):
   print 'in'
   d=cv.subtract(img,img2)
   if(d.any()!=0):
    print ' no'
   else:
    print 'same'
 else:
   print 'no'
else:
 print 'no'

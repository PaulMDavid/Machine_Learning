#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/ml/day12/index.jpeg')
f=cv.CascadeClassifier('/home/ai11/ml/day12/classifier WallClock.xml')
e=cv.CascadeClassifier('/home/ai11/Desktop/common/ML/Day12/haarcascade_eye.xml')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=f.detectMultiScale(gray,1.1)
for(x,y,w,h) in faces:
 cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
roi_color=img[y:y+h,x:x+w]
roi_gray=gray[y:y+h,x:x+h]
print(faces)
#cv.imshow('sd',img)
#cv.waitKey(0)
eyes=e.detectMultiScale(roi_gray)
for(ex,ey,ew,eh) in eyes:
 cv.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),3)
cv.imshow('sd',img)
cv.waitKey(0)

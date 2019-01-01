#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day12/car.jpg')
ret,dt=cv.threshold(img,127,255,cv.THRESH_BINARY)
print dt
cv.imshow('ds',dt)
cv.waitKey(0)
r,d=cv.threshold(img,127,155,cv.THRESH_TRUNC)
cv.imshow('sd',d)
cv.waitKey(0)
r,da=cv.threshold(img,50,255,cv.THRESH_BINARY_INV)
cv.imshow('sd',da)
cv.waitKey(0)
r,da=cv.threshold(img,50,255,cv.THRESH_TOZERO)
cv.imshow('sd',da)
cv.waitKey(0)

r,da=cv.threshold(img,50,255,cv.THRESH_TOZERO_INV)
cv.imshow('sd',da)
cv.waitKey(0)

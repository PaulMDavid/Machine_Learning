#!/usr/bin/env python
import numpy as np
import cv2 as cv
img=cv.imread('/home/ai11/Desktop/common/ML/Day11/sudoku.jpg',0)
dt=cv.blur(img,(5,5))
cv.imshow('sd',dt)
cv.waitKey(0)

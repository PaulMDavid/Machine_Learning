import cv2 as cv
import numpy as np

img2 = np.zeros((512,512,3),np.uint8)

cv.line(img2,(10,1),(500,100),(200,0,0),3)
cv.rectangle(img2,(10,1),(500,100),(0,0,200),2)
cv.imshow('im1',img2)
cv.waitKey(0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

img_ = cv2.imread(‘right.JPG’)
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img = cv2.imread(‘left.JPG’)
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift
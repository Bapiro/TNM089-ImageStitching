#import imagestitching.py
import cv2
import numpy as np
import laplacianpyramid.py as lp

images = [
    cv2.cvtColor(cv2.imread('./Images/left.jpg'), cv2.COLOR_BGR2GRAY), 
    cv2.cvtColor(cv2.imread('./Images/right.jpg'), cv2.COLOR_BGR2GRAY)
]



for img in images:
    print("hej")
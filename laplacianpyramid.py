# adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html

import cv2
import numpy as np

def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LA = []
    for la,gm in zip(lpA, gpMr):
        la = la * gm #+ lb * (1.0 - gm)
        LA.append(la)

    LB = []
    for lb,gm in zip(lpB, gpMr):
        lb = lb * gm
        LB.append(lb)

    # now reconstruct
    la_ = LA[0]
    for i in range(1,num_levels):
        la_ = cv2.pyrUp(la_)
        la_ = cv2.add(la_, LA[i])

    lb_ = LB[0]
    for i in range(1,num_levels):
        lb_ = cv2.pyrUp(lb_)
        lb_ = cv2.add(lb_, LB[i])


    return la_, lb_

#if __name__ == '__main__':
#    A = cv2.imread("./Images/right.jpg",0)
#    B = cv2.imread("./Images/left.jpg",0)
#    m = np.zeros_like(A, dtype='float32')
#    m[:,int(A.shape[1]/2):] = 1 # make the mask half-and-half
#    lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 5)
#    cv2.imwrite("lpb.png",lpb)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
# from skimage.transform import match_histograms
from sklearn.cluster import KMeans
import image_hist
import random
import math


def stitching(img_, img):
    # Convert images to greyscale
    #img_ = cv2.imread('./Images/right2.JPG')
    #img = cv2.imread('./Images/left2.JPG')

    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # Find keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    # Collect the two best matches in des1 and des2
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:
            good.append(m)
            matches = np.asarray(good)

    # Image alignment
    if len(matches[:, 0]) >= 4:
        src = np.float32(
            [kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32(
            [kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print(H)
    else:
        raise AssertionError("Cant' find enough keypoints")

    img_output = img_
    # np.uint8(image_hist.hist_match(img_, img))

    img_output2 = img

    # Stitching
    dst = cv2.warpPerspective(
        img_output, H, (img.shape[1] + img_.shape[1], img.shape[0] + 500))
    #plt.subplot(122), plt.imshow(dst), plt.title('Warped Image')
    # plt.show()
    # plt.figure()
    # dst[0:img.shape[0], 0:img.shape[1]] = img_output2

    # plt.imshow(dst)
    # plt.show()
    seamWitdh = 40
    # Taking random pixel from left/right image in a 20px width
    startLength = img.shape[1]-seamWitdh
    dst[0:img.shape[0], 0:startLength] = img_output2[0:img.shape[0], 0:startLength]
    # plt.imshow(dst)
    # plt.show()
    rnd = 0
    weightLeft = 1
    weightRight = 0.0
    step = 1/seamWitdh
    for x in range(startLength, img.shape[1]):
        # print(weightRight)
        for y in range(img.shape[0]):
            dst[y, x, 0] = (math.sqrt(
                ((weightLeft*(int(img_output2[y, x, 0])**2))+(weightRight*(int(dst[y, x, 0])**2)))))
            dst[y, x, 1] = (math.sqrt(
                ((weightLeft*(int(img_output2[y, x, 1])**2))+(weightRight*(int(dst[y, x, 1])**2)))))
            dst[y, x, 2] = (math.sqrt(
                ((weightLeft*(int(img_output2[y, x, 2])**2))+(weightRight*(int(dst[y, x, 2])**2)))))
            #rnd = random.randrange(0, 2)
            # if(rnd == 1):
            # dst[y, x] = img_output2[y, x]
            # print(1)
            # else:
            # print(0)

        weightRight += step
        weightLeft = 1-weightRight

    # blurred_img = cv2.medianBlur(dst, 3)
    # blurred_img = cv2.GaussianBlur(dst, (5, 1), 0)
    blurred_img = cv2.bilateralFilter(dst, 5, 35, 35)
    mask = np.zeros(dst.shape, "uint8")
    mask = cv2.rectangle(
        mask, (img.shape[1]-25, 0), (img.shape[1], img.shape[0]), (255, 255, 255), -1)

    out = np.where(mask == (0, 0, 0), dst, blurred_img)

    cv2.imwrite('output.jpg', dst)
    # plt.imshow(dst)
    # plt.show()

    return dst

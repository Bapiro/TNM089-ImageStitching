import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from skimage.transform import match_histograms
from sklearn.cluster import KMeans
import image_hist
# Convert images to greyscale
img_ = cv2.imread('./Images/right.JPG')
img = cv2.imread('./Images/left.JPG')


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


img_yuv = cv2.cvtColor(img_, cv2.COLOR_BGR2YUV)
img_yuv2 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

img_yuv2[:, :, 0] = cv2.equalizeHist(img_yuv2[:, :, 0])

# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
img_output2 = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)

img_output = image_hist.hist_match(img_, img)

img_output2 = img


# Stitching
dst = cv2.warpPerspective(
    img_output, H, (img.shape[1] + img_.shape[1], img.shape[0]))
plt.subplot(122), plt.imshow(dst), plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img_output2
cv2.imwrite('output.jpg', dst)
plt.imshow(dst)
plt.show()

# import imagestitching.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imagestitching
import image_hist

images = [
    # cv2.cvtColor(cv2.imread('./Images/right.jpg'), cv2.COLOR_BGR2GRAY),
    # cv2.cvtColor(cv2.imread('./Images/left.jpg'), cv2.COLOR_BGR2GRAY)
    # cv2.imread('./Images/multipleTest1.jpg'),
    # cv2.imread('./Images/multipleTest2.jpg'),
    # cv2.imread('./Images/multipleTest3.jpg'),
    # cv2.imread('./Images/multipleTest4.jpg')
    # cv2.imread('./Images/right2.jpg'),
    # cv2.imread('./Images/left2.jpg')
    cv2.imread('./Images/landscape-9.jpg'),
    cv2.imread('./Images/landscape-8.jpg'),
    cv2.imread('./Images/landscape-7.jpg'),
    cv2.imread('./Images/landscape-6.jpg'),
    cv2.imread('./Images/landscape-5.jpg'),
    cv2.imread('./Images/landscape-4.jpg'),
    cv2.imread('./Images/landscape-3.jpg'),
    cv2.imread('./Images/landscape-2.jpg'),
    cv2.imread('./Images/landscape-1.jpg'),



]

# finalimg = imagestitching.stitching(img, img)
finalimg = images[0]

imgList = []

imgList.append(finalimg)
# Color correction
counter = 0
for img in images[1:]:
    counter = counter + 1
    images[counter] = np.uint8(image_hist.hist_match(img, images[0]))

# Stitch the images together
for img in images[1:]:
    finalimg = imagestitching.stitching(finalimg, img)
    imgList.append(finalimg)

# print(len(imgList))


plt.subplot(122), plt.imshow(imgList[0]), plt.title('img 1')
plt.show()
plt.figure()

plt.subplot(122), plt.imshow(imgList[1]), plt.title('img 2')
plt.show()
plt.figure()


cv2.imwrite('test.jpg', finalimg)

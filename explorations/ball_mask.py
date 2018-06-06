import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = 'Screenshot_Wasteland.jpg'

img = cv2.imread(img_path)
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 5)
edges = cv2.Canny(gray, 100, 200)
cimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# cv2.imshow('edges', gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

#
# cv2.imshow('gray', edges)
#
# cv2.imshow('wasteland', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


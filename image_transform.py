import cv2  # opencv-python
import numpy as np
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local  # scikit-image
import imutils

# read the input image
image = cv2.imread("red.png")

# clone the original image
original_image = image.copy()

# resize using ratio (old height to the new height)
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)
#  change the color space to YUV
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# grap only the Y component
image_y = np.zeros(image_yuv.shape[0:2], np.uint8)
image_y[:, :] = image_yuv[:, :, 0]
# blur the image to reduce high frequency noises
image_blurred = cv2.GaussianBlur(image_y, (3, 3), 0)
# find edges in the image
edges = cv2.Canny(image_blurred, 50, 200, apertureSize=3)
# find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
# !! Attention !! Do not draw contours on the image at this point
# I have drawn all the contours just to show below image
# to collect all the detected polygons
polygons = []

# loop over the contours
for cnt in contours:
    # find the convex hull
    hull = cv2.convexHull(cnt)

    # compute the approx polygon and put it into polygons
    polygons.append(cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), False))
# sort polygons in desc order of contour area
sortedPoly = sorted(polygons, key=cv2.contourArea, reverse=True)

# draw points of the intersection of only the largest polyogon with red color
cv2.drawContours(image, sortedPoly[0], -1, (0, 0, 255), 5)
# get the contours of the largest polygon in the image
simplified_cnt = sortedPoly[0]

# check if the polygon has four point
if len(simplified_cnt) == 4:
	pass
    # trasform the prospective of original image
cropped_image = four_point_transform(original_image, simplified_cnt.reshape(4, 2) * ratio)
# Binarize the cropped image
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
T = threshold_local(gray_image, 11, offset=10, method="gaussian")
binarized_image = (gray_image > T).astype("uint8") * 255

# Show images
cv2.imshow("Original", original_image)
cv2.imshow("Scanned", binarized_image)
cv2.imshow("Cropped", cropped_image)
cv2.waitKey(0)

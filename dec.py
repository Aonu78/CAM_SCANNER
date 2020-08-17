import cv2
import numpy as np

framewidth = 640
framehight = 480

##################################
imgwidth = 640
imghight = 480
##################################
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, framehight)
cap.set(10, 130)


img = cv2.imread("asdf1.jpg")


def getcountours(img):
    bigest = np.array([])
    maxarea = 0
    countours, hirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnts in countours:
        area = cv2.contourArea(cnts)
        # print("Area",area)
        if area > 5000:
            # cv2.drawContours(imgcontour, cnts, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnts, True)
            # print("Area", area)
            # print(peri)
            approx = cv2.approxPolyDP(cnts, 0.02 * peri, True)
            if area > maxarea and len(approx) == 4:
                bigest = approx
                maxarea = area
    cv2.drawContours(imgcontour, bigest, -1, (255, 0, 0), 20)
    return bigest
    #         print(len(approx))
    #         objcor = len(approx)
    #         if objcor > 3: objectType = "Tri"
    #         else:objectType = "None"
    #         x, y, w, h = cv2.boundingRect(approx)
    #         cv2.rectangle(imgcontour, (x, y), (x + w, y + h), (255, 0, 0), 3)


def preProcessing(img):
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(grayimg, (5, 5), 0.5)
    imgCanny = cv2.Canny(imgblur, 55, 55)
    # kernal = np.ones((5,5))
    # imgDial = cv2.dilate(imgCanny,kernal,iterations=2)
    # imgthrs = cv2.erode(imgDial,kernal,iterations=1)
    # return imgThres
    return imgCanny


## it is for the creating an array of the images, and show them in to the single frame of images.
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getwrap(img, bigest):
    # print(bigest)
    bigest = reorder(bigest)  # if the image is geting the wronge image or revert image.
    points1 = np.float32(bigest)
    print(points1)
    points2 = np.float32([[0, 0], [framewidth, 0], [0, framehight], [framewidth, framehight]])
    matrix = cv2.getPerspectiveTransform(points1, points2)
    output = cv2.warpPerspective(img, matrix, (framewidth, framehight))
    imgcroped = output[50:output.shape[0] - 50, 50:output.shape[1] - 50]
    imgcroped = cv2.resize(output, (imgwidth, imghight))
    return imgcroped
    # pass


# it will reorder the the shape of the image to the original image, detected in the image.
# it wil crop the image to the original shape.
def reorder(mypoints):
    mypoints = mypoints.reshape((4, 2))
    mypointsnew = np.zeros((4, 1, 2), np.int32)
    add = mypoints.sum(1)
    # print("add", add)

    mypointsnew[0] = mypoints[np.argmin(add)]
    # print("mypointnew[0] should be == 374 229", mypointsnew[0])
    mypointsnew[3] = mypoints[np.argmax(add)]
    # print("mypointnew[3] should be == 528 251", mypointsnew[3])
    diff = np.diff(mypoints, axis=1)
    mypointsnew[1] = mypoints[np.argmin(diff)]
    # print("mypointnew[1] should be == 290 309", mypointsnew[1])
    mypointsnew[2] = mypoints[np.argmax(diff)]
    # print("mypointnew[2] should be == 448 332", mypointsnew[2])
    # print("mypoints== ",diff)

    return mypointsnew


while True:
    width, hight = 250, 350
    # success, img = cap.read()
    imgcontour = img.copy()
    cv2.resize(img, (imgwidth, imghight))
    imgThres = preProcessing(img)
    bigest = getcountours(imgThres)
    # test = getwrap(img, bigest)
    # cv2.imshow("result", stackImage)
    # test = getwrap(img, bigest)
    # imgarray = ([img, imgThres], [imgcontour, test])
    if bigest.size != 0:
        test = getwrap(img, bigest)
        graycolor = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
        retvel2, threshhold2 = cv2.threshold(graycolor, 12, 255, cv2.THRESH_BINARY)
        gues = cv2.adaptiveThreshold(graycolor, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 2)
        # cv2.imshow("result", gues)

        imgarray = ([img, imgThres, threshhold2], [imgcontour, test, gues])
    else:
        imgarray = ([img, imgThres], [img, img])

    stackImage = stackImages(0.4, imgarray)
    cv2.imshow("imgs", stackImage)
    # cv2.imshow("img", bigest)
    # cv2.imshow("sae", imgcontour)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

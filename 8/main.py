import cv2
import numpy as np



def imgprocess(img):
    row = img.shape[0]
    col = img.shape[1]
    newImg = np.zeros((row,col),dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            newImg[i][j]=img[i][j]+30
            if newImg[i][j]>=255:
                newImg[i][j]=255
    return newImg


img1 = cv2.imread("1.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("origin",img1)
cv2.imshow("n2",imgprocess(img1))

eq=cv2.equalizeHist(img1)
# cv2.imshow("Histogram Equalization",eq)

cv2.waitKey(0)
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math



def on_mouse(e,x,y,f,p):
    if e==cv2.EVENT_LBUTTONDOWN:
        print((x,y))


def do_math(img,fn):
    rows, cols = img.shape
    nimg = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            nimg[i][j] = fn(img[i][j])
    return nimg

img_name = 'building2.jpg'
img_rgb = cv2.imread(img_name)
img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)


ret,thresh1 = cv2.threshold(cv2.medianBlur(img,11),110,255,cv2.THRESH_BINARY)
plt.figure('bin'),plt.imshow(thresh1,cmap='gray')
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.medianBlur(th3, 5)
# plt.figure('adptive'),plt.imshow(th3,cmap='gray')
# thresh1 = cv2.medianBlur(thresh1,5)
# plt.figure('median'),plt.imshow(thresh1,cmap='gray')


k = np.ones((3, 3), np.uint8)
img_erode = cv2.erode(thresh1, k, iterations=3)
plt.figure('erode'),plt.imshow(img_erode,cmap='gray')

img_dilate = cv2.dilate(img_erode, k, iterations=3)
plt.figure('dilate'),plt.imshow(img_dilate,cmap='gray')


plt.figure('origin'),plt.imshow(img,cmap='gray')
plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


img_name = 'img.jpg'
img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)

ret,thresh1 = cv2.threshold(cv2.medianBlur(img,5),110,255,cv2.THRESH_BINARY)
plt.figure('bin'),plt.imshow(thresh1,cmap='gray')

(cnts, _) = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE
                                 )

result_cnts = []
most_possible = None
area_ratio = np.pi/4

for c in cnts:
    area = cv2.contourArea(c)
    (x, y, w, h) = cv2.boundingRect(c)
    if area<100 or w/h>1.2 or w/h<1/1.2 or abs(area/(w*h)-area_ratio)>0.1:
        continue
    result_cnts.append(c)
    if most_possible is None or area>cv2.contourArea(most_possible):
        most_possible = c
    # cv2.rectangle(img, (x, y), (x + w,y + h), (255, 255, 255), 3)

(x, y, w, h) = cv2.boundingRect(most_possible)
cv2.rectangle(img, (x, y), (x + w,y + h), (255, 255, 255), 3)

result_mask = np.zeros(img.shape)
result_mask[y:y+h,x:x+w]=1
result_img = img*result_mask
plt.figure('result'),plt.imshow(result_img,cmap='gray')
plt.figure('origin'),plt.imshow(img,cmap='gray')
plt.show()
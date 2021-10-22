import numpy as np
import cv2

img = cv2.imread("img3.jpg")

rows = img.shape[0]
cols = img.shape[1]
print(rows,cols)

def mouse_down(e,x,y,flags,params):
    if e ==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.imshow("origin",img)

pts1 = np.float32([[79,106],[75,289],[183,38],[169,345]])
pts2 = np.float32([[79,106],[79,289],[185,38],[183,345]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))
cv2.imshow("toushi",dst)


cv2.setMouseCallback("origin",mouse_down)
cv2.setMouseCallback("toushi",mouse_down)
cv2.waitKey(0)
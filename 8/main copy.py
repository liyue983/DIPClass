import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread("2.jpg",cv2.IMREAD_COLOR)
cv2.imshow("origin",img1)

(b, g, r) = cv2.split(img1)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

hist = cv2.calcHist([rH], [0], None, [256], [0, 255])


result = cv2.merge((b, g, rH))
cv2.imshow("dst", result)
cv2.imwrite("dst.jpg",result)
cv2.waitKey(0)

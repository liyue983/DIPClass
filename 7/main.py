import numpy as np
import cv2

img = cv2.imread("building.png")

rows = img.shape[0]
cols = img.shape[1]
print(rows,cols)

def mouse_down(e,x,y,flags,params):
    if e ==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

cv2.imshow("origin",img)

# pts1 = np.float32([[10,100],[200,50],[50,200]])
# pts2 = np.float32([[10,100],[200,50],[100,250]])
# M = cv2.getAffineTransform(pts1,pts2)
# dst = cv2.warpAffine(img,M,(cols,rows))
# cv2.imshow("fangshe",dst)

pts1 = np.float32([[274,178],[282,340],[407,24],[456,303]])
pts2 = np.float32([[230,250],[230,400],[407,84],[407,400]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))
cv2.imshow("toushi",dst)


cv2.setMouseCallback("origin",mouse_down)
cv2.setMouseCallback("toushi",mouse_down)
cv2.waitKey(0)
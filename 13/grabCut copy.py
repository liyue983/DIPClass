
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Step1. 加载图像
img = cv2.imread('building2.jpg')

# Step2. 创建掩模、背景图和前景图
mask = np.zeros(img.shape[:2], np.uint8)# 创建大小相同的掩模
bgdModel = np.zeros((1,65), np.float64)# 创建背景图像
fgdModel = np.zeros((1,65), np.float64)# 创建前景图像

# Step3. 初始化矩形区域
# 这个矩形必须完全包含前景
x1 = 255
y1 = 213
x2 = 369
y2 = img.shape[0]#517
rect = (x1,y1,x2-x1,y2-y1) #格式为（x, y, w, h）

# Step4. GrubCut算法，迭代9次
# mask的取值为0,1,2,3
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT) # 迭代5次

# Step5. mask中，值为2和0的统一转化为0, 1和3转化为1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img_out = img * mask2[:,:,np.newaxis] # np.newaxis 插入一个新维度，相当于将二维矩阵扩充为三维

plt.figure(),plt.subplot(121),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(122),plt.imshow(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
plt.show()
# cv2.imshow("dst", img)
# cv2.waitKey(0)
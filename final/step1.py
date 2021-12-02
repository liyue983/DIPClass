import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
plt.figure()
plt.subplot(121),plt.imshow(img,cmap='gray')

##傅里叶变换
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 将空间域转换为频率域
dft_shift = np.fft.fftshift(dft)  # 将低频部分移动到图像中心
f = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
plt.subplot(122),plt.imshow(20*np.log(f+1),cmap='gray')


##mask
mask = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        mask[i][j] = 1


##竖直大条纹
# mask[120:135, 193:199] = 0 # mask[120:135, 193:199] = 0
# mask[128:145, 153:159] = 0 # mask[128:145, 153:159] = 0
mask[120:135, 190:200] = 0# mask[120:135, 190:200] = 0
mask[128:145, 155:165] = 0# mask[128:145, 155:165] = 0
mask[120:135, 184:188] = 0
mask[128:145, 167:171] = 0
##竖直小条纹
mask[:, 80:100] = 0 # mask[110:145, 80:100] = 0
mask[:, 255:275] = 0 # mask[110:145, 255:275] = 0
##大黑纹
mask[127:131, 174:180] = 0 # mask[127:131, 174:180] = 0
mask[132:136, 171:177] = 0 # mask[132:136, 171:177] = 0

##与mask相乘
fshift = dft_shift * mask
f2 = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
f_ishift = np.fft.ifftshift(fshift) # fftshit()函数的逆函数，它将频谱图像的中心低频部分移动至左上角
img_back = cv2.idft(f_ishift)/(rows*cols) # 将频率域转化回空间域，输出是一个复数，cv2.idft()返回的是一个双通道图像
img_back2 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # idft[:,:,0]求得实部，用idft[:,:,1]求得虚部。
plt.figure()
plt.subplot(121),plt.imshow(img_back2,cmap='gray')
plt.subplot(122),plt.imshow(20*np.log(f2+1),cmap='gray')

plt.show()
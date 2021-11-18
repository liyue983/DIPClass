import cv2
import numpy as np
from matplotlib import pyplot as plt


def on_mouse(e,x,y,f,p):
    if e==cv2.EVENT_LBUTTONDOWN:
        print((x,y))
    pass

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 将空间域转换为频率域
dft_shift = np.fft.fftshift(dft)  # 将低频部分移动到图像中心
rows, cols = img.shape
# cv2.imshow("dft",cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# plt.plot()
f = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

cv2.imshow("fft",f/40000)
cv2.setMouseCallback("fft",on_mouse)
mask = np.zeros((rows, cols, 2), np.uint8)
for i in range(rows):
    for j in range(cols):
        mask[i][j] = 0

m = 25 # 越小越清晰，越大越糊
n = 40 # 越小越糊，越大越清晰
l = 25
mask[int(rows / 2 - m):int(rows / 2 + m), int(cols / 2 - l):int(cols / 2 + l)] = 1
mask[120:135, 184:188] = 0
mask[120:135, 190:200] = 0
mask[128:145, 155:165] = 0
mask[128:145, 167:171] = 0
# apply mask and inverse DFT
fshift = dft_shift * mask
cv2.imshow("fft2",cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])/40000)

f_ishift = np.fft.ifftshift(fshift) # fftshit()函数的逆函数，它将频谱图像的中心低频部分移动至左上角
img_back = cv2.idft(f_ishift) # 将频率域转化回空间域，输出是一个复数，cv2.idft()返回的是一个双通道图像
img_back2 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # idft[:,:,0]求得实部，用idft[:,:,1]求得虚部。


plt.plot()
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back2, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(20*np.log(img_back2), cmap='gray') # 归一化图像
# plt.title('normalize'), plt.xticks([]), plt.yticks([])
plt.show()
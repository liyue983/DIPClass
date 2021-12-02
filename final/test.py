import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

img = cv2.imread('test2.png',cv2.IMREAD_GRAYSCALE)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 将空间域转换为频率域
dft_shift = np.fft.fftshift(dft)  # 将低频部分移动到图像中心



img2 = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
dft2 = cv2.dft(np.float32(img2), flags=cv2.DFT_COMPLEX_OUTPUT)  # 将空间域转换为频率域
dft_shift2 = np.fft.fftshift(dft2)  # 将低频部分移动到图像中心

f = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
f2 = cv2.magnitude(dft_shift2[:, :, 0], dft_shift2[:, :, 1])

plt.figure()
plt.subplot(121),plt.imshow(20*np.log(f+1),cmap='gray')
plt.subplot(122),plt.imshow(20*np.log(f2+1),cmap='gray')
plt.show()
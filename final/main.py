import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


def on_mouse(e,x,y,f,p):
    if e==cv2.EVENT_LBUTTONDOWN:
        print((x,y))
    pass

def do_math(img,fn):
    rows, cols = img.shape
    nimg = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            nimg[i][j] = fn(img[i][j])
    return nimg

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)

dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 将空间域转换为频率域
dft_shift = np.fft.fftshift(dft)  # 将低频部分移动到图像中心
rows, cols = img.shape
# cv2.imshow("dft",cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# plt.plot()
f = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# cv2.imshow("fft",do_math(f,lambda x:math.log(abs(x)+1))/16)
# cv2.imshow("fft",20*np.log(f))
plt.figure()
plt.subplot(121),plt.imshow(20*np.log(f+1),cmap='gray')
# plt.show()

# cv2.setMouseCallback("fft",on_mouse)
mask = np.zeros((rows, cols, 2), np.float32)
for i in range(rows):
    for j in range(cols):
        mask[i][j] = 1

mask[120:135, 184:188] = 0
mask[120:135, 190:200] = 0# mask[120:135, 190:200] = 0
mask[128:145, 155:165] = 0# mask[128:145, 155:165] = 0
mask[128:145, 167:171] = 0
mask[:, 80:100] = 0 # mask[110:145, 80:100] = 0
mask[:, 255:275] = 0 # mask[110:145, 255:275] = 0

mask[126:130, 173:180] = 0
mask[131:134, 169:177] = 0
# mask[:, 174:176] = 0



m = 5
l = 5
mask[int(rows / 2 - m):int(rows / 2 + m+1), int(cols / 2 - l):int(cols / 2 + l+1)] = 0.01
k1 = 1
k2 = 1
mask[int(rows / 2 - k1):int(rows / 2 + k1+1), int(cols / 2 - k2):int(cols / 2 + k2+1)] = 1.3



# apply mask and inverse DFT
fshift = dft_shift * mask
# fshift[int(rows / 2 - m):int(rows / 2 + m), int(cols / 2 - l):int(cols / 2 + l)]=0

f2 = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
# f2[:, 80:100][20*np.log(f2[:, 80:100]+1)>150]=(math.e**(100/20))
# f2[:, 255:275][20*np.log(f2[:, 255:275]+1)>150]=(math.e**(100/20))
# cv2.imshow("fft2",do_math(f2,lambda x:math.log(abs(x)+1))/16)
plt.subplot(122),plt.imshow(20*np.log(f2+1),cmap='gray')

f_ishift = np.fft.ifftshift(fshift) # fftshit()函数的逆函数，它将频谱图像的中心低频部分移动至左上角
img_back = cv2.idft(f_ishift)/(rows*cols) # 将频率域转化回空间域，输出是一个复数，cv2.idft()返回的是一个双通道图像
img_back2 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # idft[:,:,0]求得实部，用idft[:,:,1]求得虚部。

# plt.figure()
# # plt.hist(20*np.log(f[:, 80:100]+1).ravel(),512)
# localf = f[:, 80:100].copy()
# plt.subplot(221),plt.hist(20*np.log(localf+1).ravel(),512)
# plt.subplot(223),plt.imshow(20*np.log(localf+1),cmap='gray')
# localf[20*np.log(localf+1)>140]=(math.e**(100/20))
# plt.subplot(222),plt.hist(20*np.log(localf+1).ravel(),512)
# plt.subplot(224),plt.imshow(20*np.log(localf+1),cmap='gray')

##直方图
plt.figure()
plt.hist((np.uint8(img_back2)).ravel(), 256)

##局部直方图均衡
img_back3 = np.uint8(img_back2)
clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(23,23))
img_back3 = clahe.apply(img_back3)
# plt.hist((np.uint8(img_back3)).ravel(), 256)

##图像增强
img_back4 = np.uint8(img_back2)
kernel_1 = np.float32([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
img_back4 = cv2.filter2D(img_back3,-1,kernel_1)
# cv2.imshow('back4',(img_back4))

##试一试对比度拉伸
def stretch(x):
    x1=100
    x2=170
    y1=20
    y2=245
    if x<=0:return 0 
    if x>=255:return 255
    if x<=x1:
        return x*y1/x1
    elif x1<x<=x2:
        return (x-x1)*(y2-y1)/(x2-x1)+y1
    else:
        return (x-x2)*(255-y2)/(255-x2)+y2
img_back5 = img_back2.copy()
img_back5 = do_math(img_back5,stretch)
plt.hist((np.uint8(img_back5)).ravel(), 256)
cv2.imshow('stretch',np.uint8(img_back5))


plt.figure()
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Origin'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_back2, cmap='gray')
plt.title('Output'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_back3, cmap='gray')
plt.title('Output2'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(20*np.log(img_back2), cmap='gray') # 归一化图像
# plt.title('normalize'), plt.xticks([]), plt.yticks([])
plt.show()
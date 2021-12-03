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

##直方图
plt.figure('origin hist')
plt.hist(((img_back2)).ravel(), 256)

##局部直方图均衡
img_back3 = np.uint8(img_back2)
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(5,5))
img_back3 = clahe.apply(img_back3)
plt.figure('local_hist'),plt.hist((np.uint8(img_back3)).ravel(), 256)
# plt.figure(),plt.imshow(img_back3,cmap='gray')
cv2.imshow('back3',img_back3)

##整体直方图均衡
ehist = cv2.equalizeHist(np.uint8(img_back2))
plt.figure('_hist'),plt.hist(ehist.ravel(), 256)
cv2.imshow('ehist',(ehist))



##图像增强
img_back4 = np.uint8(img_back2)
kernel_1 = np.float32([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
img_back4 = cv2.filter2D(img_back3,-1,kernel_1)
cv2.imshow('back4',(img_back4))

##对比度
def do_math(img,fn):
    rows, cols = img.shape
    nimg = np.zeros((rows, cols), np.float32)
    for i in range(rows):
        for j in range(cols):
            nimg[i][j] = fn(img[i][j])
    return nimg

def stretch(x):
    x1=70
    x2=130
    y1=20
    y2=245
    result = 0
    if x<=x1:
        result = x*y1/x1
    elif x1<x<=x2:
        result = (x-x1)*(y2-y1)/(x2-x1)+y1
    else:
        result = (x-x2)*(255-y2)/(255-x2)+y2
    if result<=0:return 0 
    if result>=255:return 255
    return result

img_back5 = img_back2.copy()
img_back5 = do_math(img_back5,stretch)
plt.figure(),plt.hist((np.uint8(img_back5)).ravel(), 256)
cv2.imshow('stretch',np.uint8(img_back5))

mask2 = np.zeros((rows, cols), np.float32)
r = 120
for i in range(rows):
    for j in range(cols):
        if((i-rows/2)**2+(j-cols/2)**2)<r*r:
            mask2[i][j] = 1
img_cut = mask2*img_back2
plt.figure('cut'),plt.imshow(img_cut,cmap='gray')

# ##中值滤波
img_median = cv2.medianBlur(np.uint8(img_back2), 5)
# plt.figure('median'),plt.imshow(img_median,cmap='gray'),plt.title('median')
# clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(25,25))
# img_local = clahe.apply(img_median)
# plt.figure('local'),plt.imshow(img_local,cmap='gray'),plt.title('local')

##二值
ret,thresh1 = cv2.threshold(np.uint8(img_back2),90,255,cv2.THRESH_BINARY)
plt.figure('bin'),plt.imshow(thresh1,cmap='gray')
th3 = cv2.adaptiveThreshold(img_median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.medianBlur(th3, 5)
plt.figure('adptive'),plt.imshow(th3,cmap='gray')

def average_g(img):
    tmp = 0
    for i in range(rows-1):
        for j in range(cols-1):
            dx = np.float32(img[i, j + 1]) - np.float32(img[i, j])
            dy = np.float32(img[i + 1, j]) - np.float32(img[i, j])
            ds = math.sqrt((dx*dx + dy*dy) / 2);
            tmp+=ds
    g = tmp / (rows*cols)
    return g

print('origin g:',average_g(img))
print('output g:',average_g(ehist))

plt.figure()
plt.subplot(121),plt.imshow(img_back2,cmap='gray')
plt.subplot(122),plt.imshow(20*np.log(f2+1),cmap='gray')

plt.show()
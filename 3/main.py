import cv2
import os
import numpy as np
from skimage.measure import compare_mse
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from skimage.measure import compare_nrmse
from skimage.measure import shannon_entropy

def load_images_from_folder(fd):
    imgs = []
    for filename in os.listdir(fd):
        img = cv2.imread(os.path.join(fd,filename))
        if (img is not None):
            imgs.append(to_gray(img))
    return imgs

def to_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def main():
    originpath = "lena.png"
    targetfolder = "dev2"
    origin = to_gray(cv2.imread(originpath))
    targetimgs = load_images_from_folder(targetfolder)

    for i in range(len(targetimgs)):
        print("="*10,"image number:",i,"="*10)
        mse = compare_mse(origin,targetimgs[i])
        print(f'MSE:{mse}')
        psnr = compare_psnr(origin,targetimgs[i])
        print(f'PSNR:{psnr}')
        ssim = compare_ssim(origin,targetimgs[i])
        print(f'SSIM:{ssim}')
        nrmse = compare_nrmse(origin,targetimgs[i])
        print(f'NRMSE:{nrmse}')
        entropy = shannon_entropy(targetimgs[i])
        print(f'Entropy:{entropy}')

    ht = np.hstack([origin,*targetimgs])
    cv2.imshow('images',ht)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
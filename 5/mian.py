import numpy as np
import cv2


class red_eye_elimate():
    def __init__(self,file_path):
        self.file_path = file_path
    
    def fillhole(self,thresh):
        h,w = thresh.shape[:]
        mask = np.zeros((h+2,w+2),np.uint8)
        holes = cv2.floodFill(thresh.copy(),mask,(0,0),255)
        s = cv2.bitwise_not(holes[1])
        full_thresh = s|thresh
        return full_thresh

    def detect_eye_and_elimate(self):
        eye = cv2.CascadeClassifier("haarcascade_eye.xml")
        eye.load("haarcascade_eye.xml")
        image = cv2.imread(self.file_path)
        cv2.imshow("orign",image)
        eyes = eye.detectMultiScale(image,1.03,20,0,(40,40))
        if len(eyes)!=0:
            for x,y,w,h in eyes:
                eye_image_bgr = image[y:y+h,x:x+w]
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                img_HSV = cv2.cvtColor(eye_image_bgr,cv2.COLOR_BGR2HSV)
                channels1 = cv2.split(img_HSV)
                mask1 = channels1[1]>100
                mask2 = channels1[0]<180
                mask3 = channels1[0]>170
                mask = mask1 & mask2 & mask3
                mask = mask.astype(np.uint8)*255
                full_mask = self.fillhole(np.uint8(mask))
                mean=np.zeros((h, w),dtype=np.uint8)
                ht2 = np.hstack((mask,full_mask,mean))
                cv2.imshow('the mask',ht2)
                eye_s_copy=channels1[1].copy()
                np.copyto(eye_s_copy, mean[:,:], where=full_mask. astype(np.bool)[:,:])
                channels1[1]=eye_s_copy
                hsv=cv2.merge(channels1)
                eye_copy = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                cv2.imshow('the mask2', eye_s_copy)
                image[y:y+h,x:x+w]=eye_copy
        return image

if __name__ == "__main__":
    file_path=r"red_eyes1.png"
    red = red_eye_elimate(file_path)
    image=red.detect_eye_and_elimate()
    cv2.imshow("sa",image)
    cv2.waitKey(0)
        
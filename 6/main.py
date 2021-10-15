import cv2

def gray2color(gr):
    if gr<50:
        return (0,0,0)
    elif 50<=gr<100:
        return (50,50,50)
    elif 100<=gr<200:
        return (220,70,0)[::-1]
    elif 200<=gr<240:
        return (240,240,90)[::-1]
    return (255,255,255)



img = cv2.imread("img.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_color = cv2.applyColorMap(img,cv2.COLORMAP_HOT)
img_color2 = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)

w = img_gray.shape[0]
h = img_gray.shape[1]
for i in range(w):
    for j in range(h):
        img_color2[i][j]=gray2color(img_gray[i][j])

cv2.imshow("gray",img_gray)
cv2.imshow("color",img_color)
cv2.imshow("color2",img_color2)
cv2.waitKey(0)




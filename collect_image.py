import cv2
import os

name = input("請輸入你的名字:  ")
path = "data/images/%s" % name
if not os.path.exists(path):
    os.mkdir(path)
print("開始拍照，一共四張，按S鍵拍照")
i = 0
cap = cv2.VideoCapture(1)
while True:
    re, img = cap.read()
    if re:
        img = cv2.resize(img, (640,480))
        cv2.imshow("photo", img)
        if i == 4:
            break
        if cv2.waitKey(1)&0xFF == ord('s'):
            cv2.imwrite(os.path.join(path, "%s_" %(name) + str(i+1) + ".jpg"), img)
            print("第%s张" %(str(i+1)))
            i += 1
    else:
        raise RuntimeError('Could not read image from camera')

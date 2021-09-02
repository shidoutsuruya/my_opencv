import cv2
import numpy as np
bs=cv2.createBackgroundSubtractorKNN(detectShadows=True)
camera=cv2.VideoCapture(0)
while True:
    ret,frame=camera.read()
    fgmask=bs.apply(frame)
    th=cv2.threshold(fgmask.copy(),thresh=244,maxval=255,type=cv2.THRESH_BINARY)[1]
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,ksize=(3,3))
    dilated=cv2.dilate(th,kernel,iterations=2)
    image,contours,hier=cv2.findContours(image=dilated,
                                        mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c)>1600:
            (x,y,w,h)=cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            cv2.putText(frame,'person',(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,color=(0,255,255),thickness=2)
    cv2.imshow('mog',fgmask)
    cv2.imshow('thresh',th)
    cv2.imshow('detection',frame)
    if cv2.waitKey(1)&0xff==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()

 

import numpy as np
import cv2
import matplotlib.pyplot as plt

cap=cv2.VideoCapture(0)
ret,frame=cap.read()
r,h,c,w=10,200,10,200
track_window=(c,r,w,h)
roi=frame[r:r+h,c:c+w]
hsv_roi=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
x=(100.,30.,32.)
y=(180.,120.,255.)
mask=cv2.inRange(hsv_roi,np.array(x),np.array(y))
roi_hist=cv2.calcHist(images=[hsv_roi],channels=[0],
                      mask=mask,histSize=[180],ranges=[0,180])
cv2.normalize(roi_hist,roi_hist,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
term_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
while True:
    ret,frame=cap.read()
    if ret==True:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst=cv2.calcBackProject(images=[hsv],channels=[0],
                                hist=roi_hist,ranges=[0,180],scale=1)
        ret,track_window=cv2.meanShift(dst,track_window,term_crit)
        x,y,w,h=track_window
        img2=cv2.rectangle(frame,(x,y),(x+w,y+h),color=255,thickness=2)
        cv2.imshow('img2',img2)
        cv2.imshow('hsv',hsv)
        cv2.imshow('dst',dst)
        k=cv2.waitKey(1)&0xff
        if k==ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()





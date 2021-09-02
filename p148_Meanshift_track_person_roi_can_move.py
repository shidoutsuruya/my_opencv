import numpy as np
import cv2

cap=cv2.VideoCapture(0)
ret,frame=cap.read()
r,h,c,w=300,200,400,300
track_window=(c,r,w,h)
roi=frame[r:r+h,c:c+w]
hsv_roi=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
x=np.array((100.,30.,32.))
y=np.array((180.,120.,255.,))
mask=cv2.inRange(hsv_roi,x,y)

roi_hist=cv2.calcHist(images=[hsv_roi],channels=[0],
                      mask=mask,histSize=[180],ranges=[0,180])
term_crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)

while True:
    ret,frame=cap.read()
    if ret == True:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst=cv2.calcBackProject(images=[hsv],channels=[0],
                                hist=roi_hist,ranges=[0,180],scale=1)
        ret,track_window=cv2.CamShift(dst,track_window,term_crit)
        pts=cv2.boxPoints(ret)
        pts=np.int0(pts)
        img2=cv2.polylines(frame,pts=[pts],isClosed=True,color=255,thickness=2)
        cv2.imshow('hsv',hsv)
        cv2.imshow('dst',dst)
        cv2.imshow('img2',img2)
        k=cv2.waitKey(1)&0xff
        if k==ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()




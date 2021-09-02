import numpy as np
import cv2
cap = cv2.VideoCapture(r'D:\ドキュメンタリー\アニメ動画\ロード・エルメロイII世の事件簿\[HorribleSubs] Lord El-Melloi II Case Files - 00 [720p].mkv')
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame',frame) 
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

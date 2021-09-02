import numpy as np
import cv2
import matplotlib.pyplot as plt
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img,center=(x,y),radius=10,color=(100,100,200),thickness=-1)

img = np.zeros((1000,1000,3))
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break
cv2.destroyAllWindows()

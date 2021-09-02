import numpy as np
import cv2
import matplotlib.pyplot as plt 

img = np.zeros((1000,1000,3))
cv2.line(img,pt1=(50,50),pt2=(511,511),color=(0,255,0),thickness=1)
cv2.rectangle(img,(384,0),(510,128),(265,0,0),-1,cv2.LINE_AA)
cv2.circle(img,center=(200,200), radius=72, color=(0,0,255), thickness=25)
c=cv2.ellipse(img,center=(256,256),axes=(100,50),angle=50,startAngle=0,
              endAngle=360,color=(255,255,255),thickness=-1)

pts = np.array([[10,5],[200,300],[70,20],[50,10],[234,34]])
pts = pts.reshape((-1,1,2))
print(pts)
cv2.polylines(img,pts=[pts],isClosed=False,color=(0,255,255),thickness=1)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,text='OpenCV',org=(300,500),fontFace= font,
           fontScale=4,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
plt.imshow(c)
plt.show()

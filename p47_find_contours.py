import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.pyrDown(cv2.imread("TEST.jpg", cv2.IMREAD_UNCHANGED))
(B,G,R)=cv2.split(img)
img=cv2.merge([R,G,B]) 
ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) ,
                           127, 255, cv2.THRESH_BINARY)
img = cv2.cvtColor(np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8),
                  cv2.COLOR_GRAY2BGR)#black background
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    #rectangle
    x,y,w,h=cv2.boundingRect(c)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #min_area_rectangle
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),3)
    #circle
    (x,y),radius=cv2.minEnclosingCircle(c)
    center=(int(x),int(y))
    radius=int(radius)
    img=cv2.circle(img,center,radius,(255,0,0),2)
#ploydp
for cnt in contours:
  epsilon = 0.01 * cv2.arcLength(cnt,True)
  approx = cv2.approxPolyDP(cnt,epsilon,True)
  hull = cv2.convexHull(cnt)
  cv2.drawContours(img, [cnt], -1, (0, 255, 255), 2) #sky blue
  cv2.drawContours(img, [approx], -1, (255, 255, 0), 2) #yellow
  cv2.drawContours(img, [hull], -1, (255, 0, 255), 2)#pink
 

 
plt.imshow(img)
plt.axis('off')
plt.show()
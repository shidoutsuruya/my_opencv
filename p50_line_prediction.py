import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r'C:\Users\max21\Desktop\tsurumaru.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,120)


minLineLength=20
maxLineGap=5
lines=cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=100,\
    minLineLength=minLineLength,maxLineGap=maxLineGap) #rho(曲线饱满值)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(edges,(x1,y1),(x2,y2),(0,0,255),2)

plt.imshow(edges,cmap='gray')
plt.axis('off')
plt.show()




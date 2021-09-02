import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_face.jpg')
img2=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_text.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
orb=cv2.ORB_create()
kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches=bf.match(des1,des2)
matches=sorted(matches,key=lambda x:x.distance)
img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],img2,flags=2)
plt.imshow(img3)
plt.show()


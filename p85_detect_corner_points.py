import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\many.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=np.float32(gray)
dst=cv2.cornerHarris(gray,2,23,0.04)
img[dst>1e-3*dst.max()]=[0,0,255]
plt.imshow(img)
plt.show()

    


import cv2 
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread(r'C:\Users\max21\Desktop\IMG_6082.jpg',0)
img=cv2.Canny(img,200,300).astype('float')
plt.imshow(img,'gray')
plt.axis('off')
plt.show()

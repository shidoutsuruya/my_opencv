import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

kernel3x3=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
kernel5x5=np.array([[-1,-1,-1,-1,-1],
                     [-1,1,2,1,-1],
                     [-1,2,4,2,-1],
                     [-1,1,2,1,-1],
                     [-1,-1,-1,-1,-1]])

img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\test.jpg',0)

k3=ndimage.convolve(img,kernel3x3)
k5=ndimage.convolve(img,kernel5x5)
blurred=cv2.GaussianBlur(img,(11,11),0)
g_hpf=img-blurred
def show(loc,image,cmap):
    plt.subplot(loc)
    plt.imshow(image,cmap)
    plt.axis('off')

plt.figure('SHOW')
show(221,img,'gray')
show(222,k3,'Blues')
show(223,k5,'Greens')
show(224,g_hpf,'Oranges')
plt.show()




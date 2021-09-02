import cv2
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\test.jpg')
gra = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gra,cmap='gray')
plt.imshow(img)
plt.show()


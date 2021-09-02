import numpy as np
import cv2
import matplotlib.pyplot as plt
for i in range(0,500):
    image = cv2.imread('pos-'+str(i)+'.pgm')
    plt.ion()
    plt.imshow(image)
    plt.pause(1)
    plt.close()

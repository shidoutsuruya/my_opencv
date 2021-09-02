import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
for i in range(0,1000):
    plt.figure(str(i))
    img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\p77_save_data\\'+str(i)+'.pgm')
    plt.ion()
    plt.imshow(img,cmap='Blues')
    plt.axis('off')   
    plt.pause(3)
    print(img)
    print(img.shape)
    plt.close(str(i))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

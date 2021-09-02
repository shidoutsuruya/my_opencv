import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
path=os.path.dirname(__file__)
picture_path=os.path.join(path,"test_picture","picture.jpg")
cascade_path=os.path.join(path,"cascades","haarcascade_frontalface_default.xml")
def detect(file):
    face_cascade=cv2.CascadeClassifier\
        (cascade_path)
    img=cv2.imread(file)
    gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    print(gray.shape)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

detect(picture_path)


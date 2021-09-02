import cv2
import numpy as np
def resize(img,scaleFactor):
    x=int(img.shape[1]*(1/scaleFactor))
    y=int(img.shape[0]*(1/scaleFactor))
    return cv2.resize(img,dsize=(x,y),interpolation=cv2.INTER_AREA)

def pyramid(image,scale=1.5,minSize=(200,80)):
    yield image
    while True:
        image=resize(image,scale)
        if image.shape[0]<minSize[1] or image.shape[1]<minSize[0]:
            break
        yield image





import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
imgpath=os.path.join(os.path.dirname(__file__),'test.jpg')
img=cv2.imread(imgpath)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

#sift=cv2.xfeatures2d.SIFT_create()
#keypoints,descriptor=sift.detectAndCompute(gray,None)
#img=cv2.drawKeypoints(image=img,
                      #outImage=img,
                      #keypoints=keypoints,
                      #flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                      #color=(255,255,0))
plt.imshow(gray)
plt.show()



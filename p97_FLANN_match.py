import cv2
import numpy as np
import matplotlib.pyplot as plt
img1=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_face.jpg')
img2=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_text.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
sift=cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams=dict(check=50)
flann=cv2.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)
matchesMask=[[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance< 0.4*n.distance:#default 0.7
        matchesMask[i]=[1,0]
drawParamas=dict(matchColor=(0,255,255),
                 singlePointColor=(255,254,0),
                 matchesMask=matchesMask,
                 flags=0)
resultImage=cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**drawParamas)
plt.imshow(resultImage)
plt.show()


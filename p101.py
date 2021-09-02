import numpy as np
import cv2
import matplotlib.pyplot as plt
MIN_MATCH_COUNT=80
path=[r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_face.jpg',
     r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_text.jpg']
def sift(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift=cv2.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(img,None)
    return kp,des,img
kp1=sift(path[0])[0]
kp2=sift(path[1])[0]
des1=sift(path[0])[1]
des2=sift(path[1])[1]

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params=dict(checks=50)
flann=cv2.FlannBasedMatcher(index_params,search_params)
matches=flann.knnMatch(des1,des2,k=2)

good=[]
for m,n in matches:
    if m.distance<0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    #find important points
    src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
    matchesMask=mask.ravel().tolist()
    h,w=sift(path[0])[2].shape
    pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)
    img1=sift(path[0])[2]
    img2=sift(path[1])[2]
    img2=cv2.polylines(img2,[np.int32(dst)],True,color=255,
                       thickness=3,lineType=cv2.LINE_AA)
else:
    print('not enough matches are found!!!{0}/{1}'.format(len(good),MIN_MATCH_COUNT))
    matchesMask=None

draw_params=dict(matchColor=(0,255,0),singlePointColor=None,
                 matchesMask=matchesMask,flags=2)
img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3)
plt.show()

    




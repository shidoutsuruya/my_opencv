from os.path import join
from os import walk
import numpy as np
import cv2
from sys import argv

folder=argv[1]
query=cv2.imread(join(folder,r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_face.jpg'))
files=[]
images=[]
descriptors=[]
for (dirpath,dirnames,filenames) in walk(folder):
    files.extend(filenames)
    for f in files:
        if f.endswith('npy') and f!='test_face.npy':
            descriptors.append(f)
        print(descriptors)

sift=cv2.xfeatures2d.SIFT_create()
query_kp,query_ds=sift.detectAndCompute(query,None)

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params=dict(checks=50)
flann=cv2.FlannBasedMatcher(index_params,search_params)

MIN_MATCH_COUNT=10
potential_culprits={}

print('start scan...')
for des in descriptors:
    print('analyzing %s for matches...'%des)
    matches=flann.knnMatch(query_ds,np.load(join(folder,des)),k=2)
    good=[]
    for m,n in matches:
        if m.distance<0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        print('%s is a match!(%d)'%(des,len(good)))
    else:
        print('%s is not a match'%des)
    potential_culprits[des]=len(good)

max_matches=None
potential_suspect=None
for culprit,matches in potential_culprits.items():
    if max_matches==None or matches>max_matches:
        max_matches=matches
        potential_suspect=culprit

print('potential suspect is %s'%potential_suspect.replace('npy','').upper())




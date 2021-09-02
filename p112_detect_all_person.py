import cv2
import numpy as np
import matplotlib.pyplot as plt
def is_inside(o,i):
    ox,oy,ow,oh=o
    ix,iy,iw,ih=i
    return ox>ix and oy>iy and ox+ow<ix+iw and oy+oh<iy+ih

def draw_person(image,person):
    x,y,w,h=person
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\test_face.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found,w=hog.detectMultiScale(img)
found_filtered=[]
for ri,r in enumerate(found):
    for qi,q in enumerate(found):
        if ri !=qi and is_inside(r,q):
            break
    else:
        found_filtered.append(r)

for person in found_filtered:
    draw_person(img,person)

plt.imshow(img)
plt.axis('off')
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
img=cv2.imread(r'C:\Users\max21\Desktop\Python\OpenCV\OpenCV\OpenCV\tsurumaru.jpg')
mask=np.zeros(img.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)

rect=(20,20,img.shape[1],img.shape[0])#(x,y,real_width,real_height)according to your picture
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
imgc=img.copy()*mask2[:,:,np.newaxis]

plt.subplot(121)
plt.imshow(cv2.cvtColor(imgc,cv2.COLOR_BGR2RGB)  )
plt.title('grabcut')
plt.axis('off')
plt.subplot(122)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title('original')
plt.axis('off')
plt.show()





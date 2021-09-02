import cv2
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
datapath=r'C:\Users\max21\Desktop\Python\OpenCV\carData\TrainImages'+'\\'
def path(cls,i):
    return r'%s%s%d.pgm'%(datapath,cls,i+1)

pos,neg='pos-','neg-'
detect=cv2.xfeatures2d.SIFT_create()
extract=cv2.xfeatures2d.SIFT_create() 

flann_params=dict(algorithm=1,trees=5)
flann=cv2.FlannBasedMatcher(flann_params,{})
bow_kmeans_trainer=cv2.BOWKMeansTrainer(40)
extract_bow=cv2.BOWImgDescriptorExtractor(extract,flann)

def extract_sift(fn):
    im=cv2.imread(fn,0)
    return extract.compute(im,detect.detect(im))[1]


for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos,i)))
    bow_kmeans_trainer.add(extract_sift(path(neg,i)))

voc=bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)

def bow_features(fn):
    im=cv2.imread(fn,0)
    return extract_bow.compute(im,detect.detect(im))
traindata,trainlabels=[],[]
for i in range(40):
    traindata.extend(bow_features(path(pos,i)))
    trainlabels.append(1)
    traindata.extend(bow_features(path(neg,i)))
    trainlabels.append(-1)

svm=cv2.ml.SVM_create()
svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))

def predict(fn):
    f=bow_features(fn)
    p=svm.predict(f)
    print(fn,'\t',p[1][0][0])
    return p
car=r'C:\Users\max21\Desktop\Python\OpenCV\car_test\test_car.jpg'
notcar=r'C:\Users\max21\Desktop\Python\OpenCV\car_test\test_not_car.jpg'
car_img=cv2.imread(car)
notcar_img=cv2.imread(notcar)
def color_convert(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

car_img=color_convert(car_img)
notcar_img=color_convert(notcar_img)
car_predict=predict(car)
not_car_predict=predict(notcar)

font=cv2.FONT_HERSHEY_SIMPLEX
if (car_predict[1][0][0]==1.0):
    cv2.putText(img=car_img,text='car detected',org=(10,30),
                fontFace=font,fontScale=1,color=(0,0,255),
                thickness=2,lineType=cv2.LINE_AA)

if(not_car_predict[1][0][0]==-1.0):
    cv2.putText(img=notcar_img,text='car not detected',
                org=(10,30),fontFace=font,fontScale=1,color=(0,0,255),
                thickness=2,lineType=cv2.LINE_AA)

plt.imshow(car_img)
plt.show()
plt.imshow(notcar_img)
plt.show()




import cv2
import numpy as np
datapath=r'C:\Users\max21\Desktop\Python\OpenCV\carData\TrainImages'
SAMPLES=499
def path(cls,i):
    return '%s\%s%d.pgm'%(datapath,cls,i+1)

def get_flann_matcher():
    flann_params=dict(algorithm=1,trees=5)
    return cv2.FlannBasedMatcher(flann_params,{})
def get_bow_extractor(extract,flann):
    return cv2.BOWImgDescriptorExtractor(extract,flann)

def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(),cv2.xfeatures2d.SIFT_create()
def extract_sift(fn,extractor,detector):
    im=cv2.imread(fn,0)
    return extractor.compute(im,detector.detect(im))[1]
def bow_features(img,extract_bow,detector):
    return extract_bow.compute(img,detector.detect(img))

def car_detector():
    pos,neg='pos-','neg-'
    detect,extract=get_extract_detect()
    matcher=get_flann_matcher()
    print('building BOWKmeansTrainer...')
    bow_kmeans_trainer=cv2.BOWKMeansTrainer(200)
    extract_bow=cv2.BOWImgDescriptorExtractor(extract,matcher)
    print('adding feature to trainer')
    for i in range(SAMPLES):
        print(i)
        bow_kmeans_trainer.add(extract_sift(path(pos,i),extract,detect))
        bow_kmeans_trainer.add(extract_sift(path(neg,i),extract,detect))
    voc=bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(voc)

    traindata,trainlabels=[],[]
    print('adding to train data')
    for i in range(SAMPLES):
        print(i)
        traindata.extend(bow_features(cv2.imread(path(pos,i),0),extract_bow,detect))
        trainlabels.append(1)
        traindata.extend(bow_features(cv2.imread(path(neg,i),0),extract_bow,detect))
        trainlabels.append(-1)

    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(traindata),cv2.ml.ROW_SAMPLE,np.array(trainlabels))
    svm.save(r"C:\Users\max21\Desktop\Python\OpenCV\car_detector\svmtest.mat")
    return svm,extract_bow



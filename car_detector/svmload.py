import cv2
import numpy as np
def svmtest(model_path,test_file,resize):
        svm = cv2.ml.SVM_load(model_path)
        test_data = []
        img = cv2.imread(test_file)
        img = cv2.resize(img,resize,interpolation=cv2.INTER_CUBIC)
        new_img = img.reshape((1,resize[0]*resize[1]*3))
        test_data .append(new_img[0])
        (ret, res) = svm.predict(np.asarray(test_data) )  
        for i,r in enumerate(res):
            text = ""
            text = text + str(int(r[0]))
            label = self.img_labels[i]
            cv2.putText(img,'result:'+text,(0,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            cv2.putText(img,'label:'+str(label),(0,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            ff = os.path.basename(test_file)
            cv2.imwrite("out"+ff,img)

svmtest(r'svmtest.mat','svm_test.jpg',(40,100))

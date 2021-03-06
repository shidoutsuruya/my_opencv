import cv2,os,sys
import numpy as np

def read_images(path, sz = None):
    c = 0
    x, y = [], []
    names=[]
    for dirname, dirnames, filenames in os.walk(path):
        names.extend(dirnames)
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if not filename.endswith('.pgm'):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)#input 200*200 picture
                    if sz is not None:
                        im = cv2.resize(im,(200,200))               
                    x.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except:
                    print("Unexpected error:",sys.exc_info()[0])
            c = c + 1
    x=np.asarray(x)
    y=np.asarray(y)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)  
    print(names)
    return [x, y],names

def face_rec(img_path):    
    names = read_images(img_path)[1]
    names=list(set(names))
    print(names)
    [x,y] = read_images(img_path)[0]
    model = cv2.face.EigenFaceRecognizer_create()
    print('start_train')
    model.train(x,y)
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
            r'C:\Users\max21\Desktop\Python\OpenCV\cascades\haarcascade_frontalface_default.xml')
    while (True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x: x+w, y: y+h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print("Label: %s,Name:%s,Confidence: %.2f" % (params[0],names[params[0]],(20000-params[1])/20000))
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLE, 1, 255, 2)
            except:
                continue
        cv2.imshow("camera",img)
        if cv2.waitKey(1000//12) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec(r'C:\Users\max21\Desktop\Python\OpenCV\p77_save_data\\')



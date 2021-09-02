import cv2
import os 
def generate(name):
    train_face=r'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\cascades\haarcascade_frontalface_default.xml'
    face_cascade=cv2.CascadeClassifier(train_face)
    train_eye=r'C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\cascades\haarcascade_eye.xml'
    eye_cascade=cv2.CascadeClassifier(train_eye)
    camera=cv2.VideoCapture(0)
    count=0
    path=r'C:\Users\max21\Desktop\Python\OpenCV\p77_save_data'+'\\'+str(name)
    try:
        os.mkdir(path)
    except:
        name=input('This folder exists this name. check your input or input alternative one.\n')
        return generate(name)
    print('create new folder!')
    while (True):
        ret,frame=camera.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            f=cv2.resize(gray[y:y+h,x:x+w],(200,200))       
            cv2.imwrite(path+'\\'+r'%s.pgm'%(str(count)),f)
            count+=1
            print('have collected:',count)
        cv2.imshow('camera',frame)
        if cv2.waitKey(5)&0xff==ord('q') or count==100:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    name=input('please input your name to start collecting your face information:\n')    
    generate(name)
    print('please focus on the lens...')
    print('information collection is over.')
                                

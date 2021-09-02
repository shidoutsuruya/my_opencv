import cv2
import numpy as np
import matplotlib.pyplot as plt

camera_capture=cv2.VideoCapture(0)
fps=30
x=int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
y=int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size=(x,y)

videowriter=cv2.VideoWriter(r'C:\Users\max21\Desktop\my_output_video.avi',
                            cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
success,frame=camera_capture.read()
num_frame_remaining=10*fps-1
while success and num_frame_remaining>0:
    videowriter.write(frame)   
    success,frame=camera_capture.read()
    cv2.imshow('frame!!!',frame)
    num_frame_remaining-=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
camera_capture.release()
cv2.destroyAllWindows()

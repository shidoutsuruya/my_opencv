import cv2
clicked=False

def onMouse(event,x,y,flags,param):
    global clicked
    if event==cv2.EVENT_LBUTTONUP:
        clicked=True

camera_capture=cv2.VideoCapture(0)
cv2.namedWindow('video')
cv2.setMouseCallback('video',onMouse)
print('showing camera feed. click window or pres any key to stop.')
success,frame=camera_capture.read()
while success and cv2.waitKey(1)==-1 and not clicked:
    cv2.imshow('video',frame)
    sucess,frame=camera_capture.read()

cv2.destroyWindow('video')
camera_capture.release()



import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\max21\Desktop\Python\OpenCV\car_detector')
from detector import car_detector,bow_features
from pyramid import pyramid
from non_maximum import non_max_suppression_fast
from sliding_window import sliding_window
 
def in_range(number, test, thresh=0.2):
  return abs(number - test) < thresh
test_image = r"C:\Users\max21\Desktop\police.jpg" 
svm, extractor = car_detector()
detect = cv2.xfeatures2d.SIFT_create()
 
w, h = 150, 80
img = cv2.imread(test_image)
 
rectangles = []
scaleFactor = 1.25
for resized in pyramid(img, scaleFactor):      
  scale = float(img.shape[1]) / float(resized.shape[1])  
  for (x, y, roi) in sliding_window(resized, 20, (w, h)):   
    if roi.shape[1] != w or roi.shape[0] != h:
      continue
    try:
      bf = bow_features(roi, extractor, detect)
      _, result = svm.predict(bf)
      a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL)  
      if result[0][0] == 1 and res[0][0] < -1.0:
        print ("Class: %d, Score: %f, a: %s" % (result[0][0], res[0][0], res))
        rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w) * scale), int((y+h) * scale) 
        rectangles.append([rx, ry, rx2, ry2, abs(res[0][0])])
    except:
      pass
   
windows = np.array(rectangles)
boxes = non_max_suppression_fast(windows, 0.25)

for (x, y, x2, y2, score) in boxes:
  print (x, y, x2, y2, score)
  cv2.rectangle(img, (int(x),int(y)),(int(x2), int(y2)),(0, 255, 0), 1)
  cv2.putText(img, "%f" % score, (int(x),int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
 
plt.imshow(img)
plt.axis('off')
plt.show()

import urllib
import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"C:\Users\ACER\SubhamDEV_Face_Detection\haarcascade_frontalface_default (1).xml")

model = load_model(r"C:\Users\ACER\SubhamDEV_Face_Detection\model1.h5")

URL = "http://192.168.1.102:8080/shot.jpg"

def get_pred_label(pred):
    labels = ["Jyoti"]
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1,100,100,1)
    img = img/255
    return img
    


ret = True
while ret:
    
    img_url = urllib.request.urlopen(URL)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
      
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (200,200),cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),2)
        
    cv2.imshow("capture",frame)
    if cv2.waitKey(1)==ord('Q'):
        break

cv2.destroyAllWindows()


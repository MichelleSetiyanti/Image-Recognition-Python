import time
import cv2
import os
from PIL import Image
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('DataSet/training.xml')
a = 0
while True:
    a = a + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facedetect.detectMultiScale(abu, 1.3, 5)
    for(x, y, w, h) in wajah:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 200), 2)
        id, conf = recognizer.predict(abu[y:y+h, x:x+w])
        # if(id == 1):
        #     id = "Michelle"
        # elif(id == 2):
        #     id = "Lionel"
        # elif(id == 3):
        #     id = "Jesika"
        # elif(id == 4):
        #     id = "Hanssen"
        # elif(id == 5):
        #     id = "Lionel"
        cv2.putText(frame, str(id), (x+40, y-10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
    cv2.imshow("face Recognition", frame)
    key = cv2.waitKey(1)
    if key == ord('a'):
        break
# print(check)
# print(frame)
video.release()
cv2.destroyAllWindows()

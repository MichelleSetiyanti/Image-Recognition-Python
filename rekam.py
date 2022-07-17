import time
import cv2
camera = 0
video = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id = input('Masukkan Id : ')
nama = input('Masukkan Nama : ')

a = 0
while True:
    a = a + 1
    check, frame = video.read()
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = facedetect.detectMultiScale(abu, 1.3, 5)
    for(x, y, w, h) in wajah:
        cv2.imwrite('DataSet/'+str(nama)+'.'+str(id)+'.' +
                    str(a)+'.jpg', abu[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 200), 2)
    cv2.imshow("face Recognition", frame)
    # key = cv2.waitKey(1)
    # if key == ord('a'):
    if(a > 29):
        break
# print(check)
# print(frame)
video.release()
cv2.destroyAllWindows()

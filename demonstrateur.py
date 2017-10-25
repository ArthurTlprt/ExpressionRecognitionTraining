import numpy as np
import cv2
from PIL import Image

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        np_face = img[y:y+h, x:x+w]

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = np_face

        pil_face = Image.fromarray(np.uint8(np_face))

        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
        pil_face.save('face_resized.png')

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

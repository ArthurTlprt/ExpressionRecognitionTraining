import numpy as np
import cv2
from PIL import Image

# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')


#camera initilization
cap = cv2.VideoCapture(0)

# we read the cam indefinitely
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # if a face is found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # subarray corresponding to the face
        np_face = img[y:y+h, x:x+w]
        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np.uint8(np_face))
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)

        pil_face.save('face_resized.png')

    cv2.imshow('img',img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

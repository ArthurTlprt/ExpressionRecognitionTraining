import numpy as np
import cv2
from PIL import Image

# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')


#camera initilization (default cam is 0)
cap = cv2.VideoCapture(0)

# we read the cam indefinitely
while 1:
    ret, img = cap.read()
    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # subarray corresponding to the face
        np_face = img[y:y+h, x:x+w]
        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np_face, 'RGB')
        # adapter la taille à notre CNN
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
        # remettre l'image et rgb
        b, g, r = pil_face.split()
        pil_face = Image.merge("RGB", (r, g, b))

        # pil_face.save('face_resized.png', format='PNG')
        
        # remettre tout en np.array
        np_face = np.asarray(pil_face, dtype=np.uint8)
        print(np_face.shape)

        #call le predict

    cv2.imshow('img',img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

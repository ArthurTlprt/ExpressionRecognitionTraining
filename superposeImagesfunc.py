import numpy as np
import cv2
from PIL import Image



def superpose(frame,image,x,y): #arrays en parametres.
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    frame[x:x+width,y:y+height]=image#[0:width, 0:height,0:3]

    return frame




# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')

smile = Image.open("smile.png")
b, g, r,t = smile.split()
smile = Image.merge("RGB", (r, g, b))
np_smile=np.asarray(smile, dtype=np.uint8)
#pour la transparence pour les png.
#np_smile=np_smile[0:np_smile[0].size, 0:np_smile[1].size,0:3]
#print(np.shape(np_smile))


#camera initilization (default cam is 0)
cap = cv2.VideoCapture(0)

# we read the cam indefinitely
while 1:
    ret, img = cap.read()
    # find the face
    superpose(img,np_smile,0,0)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        #call le predict

    cv2.imshow('img',img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

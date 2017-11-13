import numpy as np
import cv2 as cv
from PIL import Image,ImageOps
import numpy as np
import h5py
from random import shuffle as S
import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, BatchNormalization, Activation, Lambda, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
import time

mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))
std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))

#0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,7: Contempt, 8: None, 9: Uncertain, 10: No-Face
def normalize(image,mean_image,std_image):
    return np.divide((image-mean_image),std_image)

def showFineResults(preds):
    L=["Neutral","Happiness", "Sadness", "Surprise", "Anger"]
    msp=np.float32(0)
    index=0
    for i in range(5):
        if preds[0][i]>msp:
            msp=preds[0][i]
            index=i


        #cv.imwrite('picturetest.png', capture)
    return(L[index])

def superpose(frame,image,x,y): #arrays en parametres.
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    frame[x:x+width,y:y+height]=image #[0:width, 0:height,0:3]
    return frame

def testTousHappy(liste):
    bool=True
    if(len(liste)==0):
        bool=False
    for i in liste:
        if i!="Happiness":
            bool=False
    return bool


log= open('log.txt', 'w')
model=load_model('irc-cnn-009-0.642313.h5')

# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv.CascadeClassifier('face.xml')


#camera initilization (default cam is 0)
cap = cv.VideoCapture(0)
lastpicture=time.time()
number=0
# we read the cam indefinitely
while 1:

    ret, img = cap.read()
    img=cv.flip(img,1)


    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    listeEmotion=[]
    for index,(x,y,w,h) in enumerate(faces):
        #drawing rectangles
        # subarray corresponding to the face
        np_face = img[y:y+h, x:x+w]     

        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np_face, 'RGB')

        # adapter la taille à notre CNN0
        #pil_face = pil_face.crop((x,y,x+w,y+h))
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
        # remettre l'image et rgb
        #b, g, r = pil_face.split()
        #pil_face = Image.merge("RGB", (r, g, b))



        # remettre tout en np.array
        np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
        superpose(img,np_face,0,49*index)
        #prediction
        np_face =np.expand_dims(np_face, axis=0)
        preds= model.predict(normalize(np_face,mean_image,std_image))

        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        listeEmotion.append(showFineResults(preds))
        print(listeEmotion)

        cv.putText(img, showFineResults(preds), (x,y+w+int(w/12)), cv.FONT_HERSHEY_PLAIN,  w/200, (0,0,255),2)
        
        
        #print(np_face.shape)

        #call le predict

    timepicture = time.time()
    if ((testTousHappy(listeEmotion)) & (timepicture - lastpicture > 5)):
        lastpicture = timepicture
        pil_image = Image.fromarray(img, 'RGB')
        b, g, r = pil_image.split()
        pil_image = Image.merge("RGB", (r, g, b))
        pil_image.save('savepicture\pict' + str(number) + '.png')
        number += 1
    img=cv.resize(img,None,fx=1.6,fy=1.6)#imgcv2.resize(img,(2,2))
    cv.imshow('img',img)
    # kill with ctr+c
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    #print(t-time.time())

cap.release()
cv.destroyAllWindows()

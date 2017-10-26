import numpy as np
import cv2
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



#0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,7: Contempt, 8: None, 9: Uncertain, 10: No-Face

def showFineResults(preds):
    L=["Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger","Contempt", "None", "Uncertain", "No-Face"]
    for i in range(11):
        if preds[0][i]==np.float32(1):
            #print(L[i-1])
            return(L[i-1])

def superpose(frame,image,x,y): #arrays en parametres.
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    frame[x:x+width,y:y+height]=image #[0:width, 0:height,0:3]
    return frame


log= open('log.txt', 'w')
model=load_model('inception_resnet.h5')

# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')


#camera initilization (default cam is 0)
cap = cv2.VideoCapture(0)
# we read the cam indefinitely
while 1:
    ret, img = cap.read()

    img=cv2.flip(img,1)
    #print(type(img),img)
    # converting into PIL.Image object to mirror
    #pil_img = Image.fromarray(img, 'RGB')
    #pil_img=ImageOps.mirror(pil_img)
    #pil_img=pil_img.resize(( 1360,960), resample=Image.BILINEAR)
    # remettre tout en np.array
    #img = np.asarray(pil_img, dtype=np.uint8)



    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    for index,(x,y,w,h) in enumerate(faces):
        #drawing rectangles
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # subarray corresponding to the face
        np_face = img[y:y+h, x:x+w]     
                                #  /!\   pour éventuellement amélorer:  remplacer par un crop depuis pil_img?
        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np_face, 'RGB')
        # adapter la taille à notre CNN0
        #pil_face = pil_face.crop((x,y,x+w,y+h))
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
        # remettre l'image et rgb
        b, g, r = pil_face.split()
        pil_face = Image.merge("RGB", (r, g, b))
        #pil_face.save('face_resized.png')


        # remettre tout en np.array
        np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
        superpose(img,np_face,0,49*index)
        #prediction
        np_face =np.expand_dims(np_face, axis=0)
        preds= model.predict(np_face)
        cv2.putText(img, showFineResults(preds), (x,y+w+int(w/12)), cv2.FONT_HERSHEY_PLAIN,  w/200, (0,0,255),2)
        #print(np_face.shape)

        #call le predict
    cv2.imshow('img',img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

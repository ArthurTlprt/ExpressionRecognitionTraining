import numpy as np
import cv2
from PIL import Image, ImageOps
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


# 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,7: Contempt, 8: None, 9: Uncertain, 10: No-Face
##############
def traitementPredict(prediction):
    indexFin=0
    probMax=0
    for i in range(11):
        if(prediction[i]>probMax):
            probMax=prediction[i]
            indexFin=i
    return indexFin

def traitement_logo(directory, listeEmot, listeFin):
    for i in range(11):
        image= Image.open( emotDirectory+listeEmot[i])
        np_image=np.asarray(image, dtype=np.uint8)
        listeFin.append(np_image)

def superpose(frame, image, x, y):  # arrays en parametres.
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    for i in range(width):
        for j in range(height):
            if image[i][j][3] != 0:
                frame[x + i][y + j] = np.flip(image[i][j][0:3], 0)
    return frame


########################################
log = open('log.txt', 'w')
model = load_model('inception_resnet.h5')

emotDirectory="D:/Utilisateur/Documents/Isen_taff/projet_semestre_1/leroy/logo/"
listeLogo=["Neutral.png","Happiness.png", "Sadness.png", "Surprise.png", "Fear.png", "Disgust.png", "Anger.png","Contempt.png", "None.png", "Uncertain.png", "NoFace.png"]
liste_npLogo=[]
traitement_logo(emotDirectory, listeLogo, liste_npLogo)

# download this file as face.xml
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')

# camera initilization (default cam is 0)
cap = cv2.VideoCapture(0)
# we read the cam indefinitely
while 1:
    t = time.time()
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    for index, (x, y, w, h) in enumerate(faces):
        # drawing rectangles
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # subarray corresponding to the face
        np_face = img[y:y + h, x:x + w]

        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np_face, 'RGB')
        # adapter la taille à notre CNN0
        # pil_face = pil_face.crop((x,y,x+w,y+h))
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
        # remettre l'image et rgb
        b, g, r = pil_face.split()
        pil_face = Image.merge("RGB", (r, g, b))
        # pil_face.save('face_resized.png')


        # remettre tout en np.array
        np_face = np.flip(np.asarray(pil_face, dtype=np.uint8), 2)
        # prediction
        np_face = np.expand_dims(np_face, axis=0)
        preds = model.predict(np_face)
        superpose(img, liste_npLogo[traitementPredict(preds[0])], y-20, x-15)


        # print(np_face.shape)

        # call le predict
    img = cv2.resize(img, None, fx=1.6, fy=1.6)  # imgcv2.resize(img,(2,2))
    cv2.imshow('img', img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    print(t - time.time())

cap.release()
cv2.destroyAllWindows()
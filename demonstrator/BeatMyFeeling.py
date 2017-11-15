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
    return(L[index])

def preparePredict(np_face):
    # converting into PIL.Image object to resize
    pil_face = Image.fromarray(np_face, 'RGB')
    pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
    # remettre tout en np.array
    np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
    #prediction
    np_face =np.expand_dims(np_face, axis=0)
    preds= model.predict(normalize(np_face,mean_image,std_image))
    return showFineResults(preds)

def splitInTwoPieces(img):
    cv.line(img, (int(windowW/2)-1, 0), (int(windowW/2)-1, windowH), (100, 100, 100))
    cv.line(img, (int(windowW/2), 0), (int(windowW/2), windowH), (100, 100, 100))
    cv.line(img, (int(windowW/2)+1, 0), (int(windowW/2)+1, windowH), (100, 100, 100))
    img1 = img[:,:int(windowW/2),:]
    img2 = img[:,int(windowW/2):,:]
    return img1, img2

def writePlayer(img):
    cv.putText(img,"Player 1", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv.putText(img,"Player 2", (int(windowW/2)+100,100), cv.FONT_HERSHEY_SIMPLEX, 1, 255)

def writeFeeling(img, feeling1, feeling2):
    print(feeling1)
    cv.putText(img,feeling1, (100,200), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv.putText(img,feeling2, (int(windowW/2)+100,200), cv.FONT_HERSHEY_SIMPLEX, 1, 255)

def play(img):

    img1, img2 = splitInTwoPieces(img)
    writePlayer(img)

    face1 = face_cascade.detectMultiScale(img1, 1.3, 5)
    face2 = face_cascade.detectMultiScale(img2, 1.3, 5)

    feeling1 = feeling2 = "No Face"

    try:
        (x,y,w,h) = face1[0]
        feeling1 = preparePredict(img1[y:y+h, x:x+w])
    except:
        pass

    try:
        (x,y,w,h) = face2[0]
        feeling2 = preparePredict(img2[y:y+h, x:x+w])
    except:
        pass

    writeFeeling(img, feeling1, feeling2)

    return img

if __name__ == "__main__":
    # initilization
    mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))
    std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))
    model=load_model('irc-cnn-009-0.642313.h5')
    face_cascade = cv.CascadeClassifier('face.xml')
    windowH=0
    windowW=0

    cap = cv.VideoCapture(0)
    # we read the cam indefinitely
    while 1:
        ret, img = cap.read()
        img = cv.flip(img,1)
        windowH = img.shape[0]
        windowW = img.shape[1]
        img = play(img)

        cv.imshow('img',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

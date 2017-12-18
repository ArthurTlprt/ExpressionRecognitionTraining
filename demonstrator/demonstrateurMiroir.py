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
import time
from socketIO_client import SocketIO, LoggingNamespace
from operator import itemgetter


s = SocketIO('localhost', 3000, LoggingNamespace)

#import webbrowser
#tentative d'implémenter une moyenne sur les prédictions pour tous les visages.

mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))
std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))



#url='http://localhost:8080/happy'
#webbrowser.open(url,new=0)
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
    return(L[index])

def superpose(frame,image,x,y): #arrays en parametres.
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    frame[x:x+width,y:y+height]=image #[0:width, 0:height,0:3]
    return frame

log= open('log.txt', 'w')
model=load_model('irc-cnn-009-0.642313.h5')

# download this file as face.xml
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('face.xml')
frameNumber=5 #
#predictions=[]
visages=[]
predsSum=[[0,0,0,0,0]]
visageIndex=0
#camera initilization (default cam is 0)
try:
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)#sets the resolution; for the hd cam.
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    ret, img = cap.read()
    img=cv2.flip(img,1)
    width=np.shape(img)[1]
    height =np.shape(img)[0]
except:
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    img=cv2.flip(img,1)
    width=np.shape(img)[1]
    height =np.shape(img)[0]
# we read the cam indefinitely


while 1:
    t=time.time()
    ret, img = cap.read()
    img=cv2.flip(img,1)
    unsortedFeelings=[]
    feelings=[]

    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)


    for index,(x,y,w,h) in enumerate(faces):
        if visages==[]:
            visages.append([x,y,w,h,[],0])
            visageIndex=len(visages)-1
            for i in range(frameNumber):
                visages[0][4].append([[0,0,0,0,0]])

        for i,visage in enumerate(visages): #si pas de visage au début, on ne passe pas dans la boucle?
            if abs(x-visage[0])<w/10 and abs(y-visage[1]<h/10):
                #ceci est un visage qui a été repéré à la frame précédente, on en repère l'index.
                visageIndex=i
            else:
                #sinon on ajoute le visage:
                visages.append([x,y,w,h,[],0]) #prediction=visage[4]/index=visage[5]
                visageIndex=len(visages)-1
                for j in range(frameNumber):
                    visages[len(visages)-1][4].append([[0,0,0,0,0]])
                break
        #drawing rectangles
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # subarray corresponding to the face
        np_face = img[y:y+h, x:x+w]
        pil_face = Image.fromarray(np_face, 'RGB')
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)



        # remettre tout en np.array
        np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
        superpose(img,np_face,0,49*index)
        #prediction
        np_face =np.expand_dims(np_face, axis=0)
        preds= model.predict(normalize(np_face,mean_image,std_image))

        visages[visageIndex][4][visages[visageIndex][5]]=preds
        if visages[visageIndex][5]==(frameNumber-1):
            visages[visageIndex][5]=0
        else: visages[visageIndex][5]+=1
        for prediction in visages[visageIndex][4]:
            predsSum=predsSum+prediction
        predsMean=predsSum/frameNumber
        predsSum=[[0,0,0,0,0]]
        cv2.putText(img, showFineResults(predsMean), (x,y+w+int(w/12)), cv2.FONT_HERSHEY_PLAIN,  w/200, (0,0,255),2)
        unsortedFeelings.append((showFineResults(predsMean),visages[visageIndex][2]))
    unsortedFeelings=sorted(unsortedFeelings, key=itemgetter(1),reverse=True)
    for prediction in unsortedFeelings:
    	feelings.append(prediction[0])

    img=cv2.resize(img,None,fx=0.5,fy=0.5)
    cv2.imshow('img',img)
    s.emit('new feeling',  feelings)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

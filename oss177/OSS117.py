import numpy as np
import cv2
from PIL import Image,ImageOps
import numpy as np
import h5py
from random import shuffle as S
import skvideo.io
import sys
#import keras
#from keras.preprocessing import image
#from keras import backend as K
#from keras.models import Sequential, load_model
#from keras.layers import Conv2D, BatchNormalization, Activation, Lambda, MaxPooling2D, Flatten, Dense, Dropout
#from keras import optimizers
#from keras.callbacks import ModelCheckpoint, CSVLogger
import time




putin = Image.open("OSS117.png")
np_putin1=np.asarray(putin, dtype=np.uint8)
np_putin2=np.copy(np_putin1)
width=np.shape(np_putin2)[0]
height =np.shape(np_putin2)[1]
for i in range (width):
    for j in range (height):
        a=np_putin2[i][j][0]
        np_putin2[i][j][0]=np_putin2[i][j][2]
        np_putin2[i][j][2]=a
putin_size=200


def superpose(frame,image,x,y,TransparencyLeft,TransparencyRight):
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    for i in range(len(TransparencyLeft)):
        for left in TransparencyLeft:
            for right in TransparencyRight:
                print(left,right)
                print(np.shape(image[left[0]:right[0]][left[1]+i][0:3]))
                print(np.shape(frame[x:x+abs(right[0]-left[0])][y:y+i]))
                print(abs(right[0]-left[0]),y+i)
                frame[x:x+abs(right[0]-left[0])][y:y+i]=np.flip(image[left[0]:right[0]][1][0:3],0)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


face_cascade = cv2.CascadeClassifier('face.xml')

def showFineResults(preds):
    L=["Neutral","Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger","Contempt", "None", "Uncertain", "No-Face"]
    msp=np.float32(0)
    index=0
    for i in range(11):
        if preds[0][i]>msp:
            msp=preds[0][i]
            index=i
    return(L[index])


face_cascade = cv2.CascadeClassifier('face.xml')

cap = cv2.VideoCapture(0)
while 1:
    #t=time.time()
    ret, img = cap.read()
    img=cv2.flip(img,1)



    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for index,(x,y,w,h) in enumerate(faces):

        np_putin=np.copy(np_putin2)

        np_putin=cv2.resize(np_putin,None,fx=w/putin_size,fy=h/putin_size)
        #putin_size*=w/putin_size
        t=time.time()
        overlay_image_alpha(img,
                    np_putin[:, :, 0:3],
                    (x-int((w/putin_size)*50),y-int((w/putin_size)*50)),
                    np_putin[:, :, 3] / 255.0)
        #superpose(img,np_putin,y-int((w/putin_size)*50),x-int((w/putin_size)*50),vladTransparencyLeft,vladTransparencyRight)
        print(time.time()-t)
    #img=cv2.resize(img,None,fx=1.6,fy=1.6)
    cv2.imshow('img',img)
    # kill with ctr+c
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    #print(t-time.time())

cap.release()
cv2.destroyAllWindows()

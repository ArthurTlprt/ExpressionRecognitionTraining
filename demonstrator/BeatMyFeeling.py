import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
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


# idée passer une vidéo et devoir imiter les expressions faciales de l'acteur

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

def getScore(np_face, feeling_id):
    # converting into PIL.Image object to resize
    pil_face = Image.fromarray(np_face, 'RGB')
    pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)
    # remettre tout en np.array
    np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
    #prediction
    np_face =np.expand_dims(np_face, axis=0)
    preds= model.predict(normalize(np_face,mean_image,std_image))
    return preds[0][feeling_id+1]

def writeScores(img, score1, score2):
    color1 = (int(255*(1-score1)), int(255*score1), 0)
    color2 = (int(255*(1-score2)), int(255*score2), 0)
    img = write( (100,130), img, str(int(score1*100)), color1, 40)
    img = write( (int(windowW/2)+100,130), img, str(int(score2*100)), color2, 40)
    return img

def writePoints(img, points1, points2):
    text1 = str(int(points1*10.0)) + "pts"
    text2 = str(int(points2*10.0)) + "pts"
    img = write( (160,130), img, text1, (255, 255, 255), 40)
    img = write( (int(windowW/2)+160,130), img, text2, (255, 255, 255), 40)
    return img

def splitInTwoPieces(img):
    cv.line(img, (0,59), (windowW,59) , (100, 100, 100))
    cv.line(img, (0,60), (windowW,60) , (100, 100, 100))
    cv.line(img, (0,61), (windowW,61) , (100, 100, 100))
    cv.line(img, (int(windowW/2)-1, 61), (int(windowW/2)-1, windowH), (100, 100, 100))
    cv.line(img, (int(windowW/2), 61), (int(windowW/2), windowH), (100, 100, 100))
    cv.line(img, (int(windowW/2)+1, 61), (int(windowW/2)+1, windowH), (100, 100, 100))
    img1 = img[:,:int(windowW/2),:]
    img2 = img[:,int(windowW/2):,:]
    return img1, img2

def writePlayer(img):
    img = write( (100,100), img, "Player 1", (255, 255, 255), 40)
    img = write( (int(windowW/2)+100,100), img, "Player 2", (255, 255, 255), 40)
    return img

def writeFeelingToDo(img, feeling_to_do):
    img = write( (int(windowW/4),0), img, feeling_to_do, (255, 255, 255), 50)
    return img


def act(img, feeling_id, feelings_mean_player1, feelings_mean_player2):
    global points1, points2
    img1, img2 = splitInTwoPieces(img)
    img = writePlayer(img)
    score1 = score2 = 0

    face1 = face_cascade.detectMultiScale(img1, 1.3, 5)
    face2 = face_cascade.detectMultiScale(img2, 1.3, 5)

    try:
        (x,y,w,h) = face1[0]
        score1 = getScore(img1[y:y+h, x:x+w], feeling_id)
        feelings_mean_player1[feeling_id].append(score1)
    except:
        pass

    try:
        (x,y,w,h) = face2[0]
        score2 = getScore(img2[y:y+h, x:x+w], feeling_id)
        feelings_mean_player2[feeling_id].append(score2)
    except:
        pass

    img = writeScores(img, score1, score2)
    points1 += score1
    points2 += score2
    img = writePoints(img, points1, points2)

    feelings_to_do = ["happy", "sad", "surprised", "anger"]
    img = writeFeelingToDo(img, 'Do '+feelings_to_do[feeling_id]+
    ' face during '+ str(time_remaining)+' sec')
    return img, feelings_mean_player1, feelings_mean_player2

def play(img, feeling_id, time_remaining, feelings_mean_player1, feelings_mean_player2, winner):
    global points1, points2
    try:
        img, feelings_mean_player1, feelings_mean_player2 = act(img, feeling_id, feelings_mean_player1, feelings_mean_player2)
    except:
        average = []
        if feeling_id == 4:
            score1 = sum([sum(feeling) for feeling in feelings_mean_player1])
            score2 = sum([sum(feeling) for feeling in feelings_mean_player2])
            if score1 > score2:
                winner = "Player1"
            else:
                winner = "Player2"
        if feeling_id > 3:
            img = writePoints(img, points1, points2)
            img = writeFeelingToDo(img, winner + ' win!!!')
    return img, winner

def write(xy, img, text, color, size):
    cv2_im_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    # Pass the image to PIL
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    # use a truetype font
    font = ImageFont.truetype("fonts/Sansation_Regular.ttf", size)
    # Draw the text
    draw.text(xy, text, font=font, fill=color)
    img= cv.cvtColor(np.array(pil_im), cv.COLOR_RGB2BGR)
    return img

if __name__ == "__main__":
    # initilization
    mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))
    std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))
    model=load_model('irc-cnn-009-0.642313.h5')
    face_cascade = cv.CascadeClassifier('face.xml')
    feelings_mean_player1 = [[], [], [], []]
    feelings_mean_player2 = [[], [], [], []]
    points1 = points2 = 0
    winner = ""
    windowH=0
    windowW=0
    cap = cv.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,1000)

    # we read the cam indefinitely
    time_start = time.time()
    feeling_duration = 6
    game_duration = 4*feeling_duration # 4 feelings, 3 seconds per feeling
    while 1:
        feeling_id = int(int(time.time()-time_start) / feeling_duration)
        time_remaining = feeling_duration-int(time.time()-time_start)%feeling_duration
        ret, img = cap.read()
        img = cv.flip(img,1)

        windowH = img.shape[0]
        windowW = img.shape[1]
        img, winner = play(img, feeling_id, time_remaining,
        feelings_mean_player1, feelings_mean_player2, winner)

        cv.imshow('img',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv.destroyAllWindows()

import numpy as np
import cv2 as cv
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

# Ce code prend en compte une moyenne de valeurs pour les visages ET prend des photo si tout le monde sourit.

mean_image = image.img_to_array(image.load_img("mean_image.png", target_size=(49, 49)))
std_image = image.img_to_array(image.load_img("std_image.png", target_size=(49, 49)))
flash_image = Image.open("flash.png")


# 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,7: Contempt, 8: None, 9: Uncertain, 10: No-Face
def normalize(image, mean_image, std_image):
    return np.divide((image - mean_image), std_image)


def showFineResults(preds):
    L = ["Neutral", "Happiness", "Sadness", "Surprise", "Anger"]
    msp = np.float32(0)
    index = 0
    for i in range(5):
        if preds[0][i] > msp:
            msp = preds[0][i]
            index = i


            # cv.imwrite('picturetest.png', capture)
    return (L[index])


def superpose(frame, image, x, y):  # arrays en parametres.
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    frame[x:x + width, y:y + height] = image  # [0:width, 0:height,0:3]
    return frame


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
    # img_overlay=np.flip(img,2)
    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def testTousHappy(liste):
    bool = True
    if (len(liste) == 0):
        bool = False
    for i in liste:
        if i != "Happiness":
            bool = False
    return bool


def traitement_logo(directory, listeChiffre, listeFin):
    for i in range(3):
        image = Image.open(directory + listeChiffre[i])
        np_image = np.asarray(image, dtype=np.uint8)
        listeFin.append(np_image)


log = open('log.txt', 'w')
model = load_model('irc-cnn-009-0.642313.h5')

# download this file as face.xml
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv.CascadeClassifier('face.xml')

# initialisation des variables pour la moyenne.
frapsNumber = 5  # nombre de fraps prises en compte pour la moyenne. 5 semble fonctionner correctement à notre fréquence de rafraichissement.
visages = []
predsSum = [[0, 0, 0, 0, 0]]
visageIndex = 0

# camera initilization (default cam is 0, 1 for USB camera)
# cap = cv.VideoCapture(0)
try:
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # sets the resolution; for the hd cam.
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    ret, img = cap.read()
    img = cv.flip(img, 1)
    width = np.shape(img)[1]
    height = np.shape(img)[0]
except:
    cap = cv.VideoCapture(0)
    ret, img = cap.read()
    img = cv.flip(img, 1)
    width = np.shape(img)[1]
    height = np.shape(img)[0]

lastpicture = time.time()
number = 0

flash = 0  # variable suivant le nombre de frame du flash.

# variables servant au timer de 3 secondes
chiffreDirectory = "images_chiffre/"
listeLogo = ["3.png", "2.png", "1.png"]
liste_npChiffre = []
traitement_logo(chiffreDirectory, listeLogo, liste_npChiffre)
time_tampon = 0

np_flash = np.zeros((height, width, 4), dtype=np.uint8) + 255
nbJoueur=int(input("Entrer le nombre de joueur"))
nbManche=int(input("Entrer le nombre de manche pour cette partie:"))
listeScore=[]
# we read the cam indefinitely
for manche in range(nbManche):
while 1:


    ret, img = cap.read()
    img = cv.flip(img, 1)

    # find the face
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # if a face is found
    listeEmotion = []
    listeScore=[0]
    for index, (x, y, w, h) in enumerate(faces):
        stillThere = 0
        # drawing rectangles
        # subarray corresponding to the face

        if visages == []:
            visages.append([x, y, w, h, [], 0])  # prediction=visage[4]/index=visage[5]
            visageIndex = len(visages) - 1
            for i in range(frapsNumber):
                visages[0][4].append([[0, 0, 0, 0, 0]])
        # print(visages)
        else:
            for i, visage in enumerate(visages):  # si pas de visage au début, on ne passe pas dans la boucle?
                if abs(x - visage[0]) < w / 2 and abs(y - visage[1] < h / 2):
                    # ceci est un visage qui a été repéré à la frame précédente, on en repère l'index.
                    visageIndex = i
                    visage[0], visage[1], visage[2], visage[3] = x, y, w, h
                    # print("on passe dans le if...")
                    stillThere = 1
            if stillThere == 0:
                # sinon on ajoute le visage:
                # print("on passe dans le else...")
                visages.append([x, y, w, h, [], 0])  # prediction=visage[4]/index=visage[5]
                visageIndex = len(visages) - 1
                for j in range(frapsNumber):
                    visages[len(visages) - 1][4].append([[0, 0, 0, 0, 0]])
                break

        np_face = img[y:y + h, x:x + w]

        # converting into PIL.Image object to resize
        pil_face = Image.fromarray(np_face, 'RGB')
        pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)

        # remettre tout en np.array
        np_face = np.flip(np.asarray(pil_face, dtype=np.uint8), 2)
        superpose(img, np_face, 0, 49 * index)
        # prediction
        np_face = np.expand_dims(np_face, axis=0)
        preds = model.predict(normalize(np_face, mean_image, std_image))

        visages[visageIndex][4][visages[visageIndex][5]] = preds
        if visages[visageIndex][5] == (frapsNumber - 1):
            visages[visageIndex][5] = 0
        else:
            visages[visageIndex][5] += 1
        for prediction in visages[visageIndex][4]:
            predsSum = predsSum + prediction
        predsMean = predsSum / frapsNumber
        # print (predsMean)
        predsSum = [[0, 0, 0, 0, 0]]

        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        listeEmotion.append(showFineResults(predsMean))
        listeScore.append(predsMean[0][1])
        # print(listeEmotion)

        cv.putText(img, showFineResults(predsMean), (x, y + w + int(w / 12)), cv.FONT_HERSHEY_PLAIN, w / 200,
                   (0, 0, 255), 2)

        # print(np_face.shape)

        # call le predict

    timepicture = time.time()
    print("voici la liste", listeScore, "le max est", max(listeScore))
    if ((testTousHappy(listeEmotion)) & (timepicture - lastpicture > 5)):
        if (time_tampon == 0):
            time_tampon = time.time()
        if (time.time() - time_tampon < 1):
            overlay_image_alpha(img, liste_npChiffre[0][:, :, :3], (int(width / 2), int(height / 2)),
                                liste_npChiffre[0][:, :, 3] / 255.0)
            # superpose(img, liste_npChiffre[0], 0,0)
        elif ((time.time() - time_tampon > 1) & (time.time() - time_tampon < 2)):
            overlay_image_alpha(img, liste_npChiffre[1][:, :, :3], (int(width / 2), int(height / 2)),
                                liste_npChiffre[1][:, :, 3] / 255.0)
            # superpose(img, liste_npChiffre[1], 0, 0)
        elif ((time.time() - time_tampon > 2) & (time.time() - time_tampon < 3)):
            overlay_image_alpha(img, liste_npChiffre[2][:, :, :3], (int(width / 2), int(height / 2)),
                                liste_npChiffre[2][:, :, 3] / 255.0)
            # superpose(img, liste_npChiffre[2], 0, 0)
        elif (time.time() - time_tampon > 3):
            lastpicture = timepicture
            pil_image = Image.fromarray(img, 'RGB')
            b, g, r = pil_image.split()
            pil_image = Image.merge("RGB", (r, g, b))
            pil_image.save('savepicture/pict' + str(number) + '.png')
            number += 1
            time_tampon = 0
            flash = 3
            overlay_image_alpha(img,
                                np_flash[:, :, 0:3],
                                (0, 0),
                                np_flash[:, :, 3] / 255.0)
    elif (testTousHappy(listeEmotion) == False):
        time_tampon = 0

    if flash != 0:
        flash = flash - 1
        overlay_image_alpha(img,
                            np_flash[:, :, 0:3],
                            (0, 0),
                            np_flash[:, :, 3] / 255.00 + flash / 3)
    # img=cv.resize(img,None,fx=1.6,fy=1.6)#imgcv2.resize(img,(2,2))
    cv.imshow('img', img)
    # kill with ctr+c
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
        # print(t-time.time())

cap.release()
cv.destroyAllWindows()
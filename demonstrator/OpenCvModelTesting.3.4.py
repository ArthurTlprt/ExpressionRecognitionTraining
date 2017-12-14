import numpy as np
import cv2 as cv
from PIL import ImageFont, ImageDraw, Image
import h5py
from random import shuffle as S
from keras.preprocessing import image
from keras.models import  load_model
import time

#______________________________________________________________________fonctions________________________________________________________________________________

feelings_to_colors = np.array(
    [[255, 255, 255],
    [10, 10, 255],
    [0, 0, 0],
    [10, 255, 10],
    [255, 10, 10]], dtype='uint8')


def normalize(image,mean_image,std_image): #normalise l'image des visages avant la prédiction pour coincider avec le modèle.
    return np.divide((image-mean_image),std_image)

def showFineResults(preds): #on transforme les données des prédictions en String correspondant
    L=["Normal","Happy", "Sad", "Surprised", "Angry"]
    msp=np.float32(0)
    index=0
    for i in range(5):
        if preds[0][i]>msp:
            msp=preds[0][i]
            index=i
    return(L[index])

def maxpred(preds): #on transforme les données des prédictions en String correspondant
    msp=np.float32(0)
    for i in range(5):
        if preds[0][i]>msp:
            msp=preds[0][i]
    return(msp)

def superpose(frame,image,x,y): #arrays en parametres.
    width=np.shape(image)[0]
    height =np.shape(image)[1]
    frame[x:x+width,y:y+height]=image #[0:width, 0:height,0:3]
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
    #img_overlay=np.flip(img,2)
    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def testTousHappy(liste):
    bool=True
    if(len(liste)==0):
        bool=False
    for i in liste:
        if i!="Happiness":
            bool=False
    return bool

def traitement_logo(directory, listeChiffre, listeFin):
    for i in range(3):
        image= Image.open( directory+listeChiffre[i])
        np_image=np.asarray(image, dtype=np.uint8)
        listeFin.append(np_image)

def lissage(visages,visageIndex,frameNumber,x,y,w,h):
    stillThere=0
    if visages==[]:
        visages.append([x,y,w,h,[],0]) #prediction=visage[4]/index=visage[5]
        visageIndex=len(visages)-1
        for i in range(frameNumber):
            visages[0][4].append([[0,0,0,0,0]])
    else:
        for i,visage in enumerate(visages): #si pas de visage au début, on ne passe pas dans la boucle?
            if abs(x-visage[0])<w/2 and abs(y-visage[1]<h/2):
                #ceci est un visage qui a été repéré à la frame précédente, on en repère l'index.
                visageIndex=i
                visage[0],visage[1],visage[2],visage[3]=x,y,w,h
                stillThere=1
        if stillThere==0:
            #sinon on ajoute le visage:
            visages.append([x,y,w,h,[],0]) #prediction=visage[4]/index=visage[5]
            visageIndex=len(visages)-1
            for j in range(frameNumber):
                visages[len(visages)-1][4].append([[0,0,0,0,0]])
    return visages, visageIndex

def photo(img,width,height,np_flash,listeEmotion,liste_npChiffre,lastpicture,time_tampon,number,flash): #on affiche le timer, puis si 3 secondes se sont passées avec le sourire, on prend la photo.
    timepicture = time.time()
    if ((testTousHappy(listeEmotion)) & (timepicture - lastpicture > 5)):
        if(time_tampon==0):
            time_tampon=time.time()
        if(time.time()-time_tampon<1):
            overlay_image_alpha(img, liste_npChiffre[0][:,:,:3], (int(width/2),int(height/2)),liste_npChiffre[0][:,:,3]/255.0 )
            #superpose(img, liste_npChiffre[0], 0,0)
        elif((time.time()-time_tampon>1) & (time.time()-time_tampon<2) ):
            overlay_image_alpha(img, liste_npChiffre[1][:,:,:3], (int(width/2),int(height/2)),liste_npChiffre[1][:,:,3]/255.0 )
            #superpose(img, liste_npChiffre[1], 0, 0)
        elif((time.time() -time_tampon> 2) & (time.time()-time_tampon< 3)):
            overlay_image_alpha(img, liste_npChiffre[2][:, :, :3], (int(width/2),int(height/2)), liste_npChiffre[2][:, :, 3] / 255.0)
            #superpose(img, liste_npChiffre[2], 0, 0)
        elif(time.time()-time_tampon> 3):
            lastpicture = timepicture
            pil_image = Image.fromarray(img, 'RGB')
            b, g, r = pil_image.split()
            pil_image = Image.merge("RGB", (r, g, b))
            pil_image.save('savepicture\pict' + str(number) + '.png')
            number += 1
            time_tampon=0
            flash=3
            overlay_image_alpha(img,
                        np_flash[:, :, 0:3],
                        (0,0),
                        np_flash[:, :, 3] / 255.0)

    elif(testTousHappy(listeEmotion)==False):
        time_tampon=0

    if flash!=0:
        overlay_image_alpha(img,
                    np_flash[:, :, 0:3],
                    (0,0),
                    np_flash[:, :, 3] / 255.00+flash/3  )
        flash=flash-1
    return(number,flash,lastpicture,time_tampon)

def prediction(np_face,mean_image,std_image,visages,visageIndex,frameNumber):
    predsSum=[[0,0,0,0,0]]
    np_face =np.expand_dims(np_face, axis=0)
    preds= model.predict(normalize(np_face,mean_image,std_image))

    visages[visageIndex][4][visages[visageIndex][5]]=preds
    if visages[visageIndex][5]==(frameNumber-1):
        visages[visageIndex][5]=0
    else: visages[visageIndex][5]+=1
    for prediction in visages[visageIndex][4]:
        predsSum=predsSum+prediction
    predsMean=predsSum/frameNumber

    return predsMean

def traitement_generique(directory, listeChiffre, listeFin, width, height):
    for i in range(3):
        image= Image.open( directory+listeChiffre[i])
        image=image.resize((width,height), resample=Image.BILINEAR)
        np_image=np.asarray(image, dtype=np.uint8)
        tampon=np.copy(np_image)
        np_image=tampon
        for i in range (height):
            for j in range (width):
                a=np_image[i][j][0]
                np_image[i][j][0]=np_image[i][j][2]
                np_image[i][j][2]=a

        listeFin.append(np_image)

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
#________________________________________________________Initialisation des variables______________________________________________________________
mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))#on charge l'image moyenne et l'écart type (pour la normalisation)
std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))
model=load_model('irc-cnn-009-0.642313.h5') #on charge le modèle
# download this file as face.xml : https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv.CascadeClassifier('face.xml') #on charge le modèle open_cv qui détecte les visages.
#camera initilization (dans l'ordre des périphériques de type "caméra", on privilégie les caméras branchées.)
try:
    cap = cv.VideoCapture(1)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)#sets the resolution; for the hd cam.
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)
    ret, img = cap.read()
    img=cv.flip(img,1)
    width=np.shape(img)[1]
    height =np.shape(img)[0]
except:
    cap = cv.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    ret, img = cap.read()
    img=cv.flip(img,1)
    width=np.shape(img)[1]
    height =np.shape(img)[0]
#initialisation des variables pour le lissage.
frameNumber=5 #nombre de frames prises en compte pour la moyenne. 5 semble fonctionner correctement à notre fréquence de rafraichissement.
visages=[]

visageIndex=0
#variables servant au timer de 3 secondes/à la prise de photo/au flash.
lastpicture=time.time() #pour le timer de 5 secondes entre deux photos.
number=0 #pour nommer les photos enregistrées.
flash=0 #pour le flash: ça fonctionne comme une "fondue" d'une image blanche superposée au fond.
chiffreDirectory="images_chiffre/"
listeLogo=["3.png", "2.png", "1.png"]
liste_npChiffre=[]
generiqueDirectory="imagestexte/"
listeGenerique=["Bonjour!.png", "CeMiroirDévoileVosEmotions.png", "Souriez!.png"]
liste_npGenerique=[]
traitement_generique(generiqueDirectory, listeGenerique, liste_npGenerique, width, height)
traitement_logo(chiffreDirectory, listeLogo, liste_npChiffre) #charge les images, les prend
time_tampon=0
tempsDebut=time.time()
#création de la matrice blanche servant de flash lors de la prise de photographie.
np_flash=np.zeros((height,width, 4),dtype=np.uint8)+255

# lecture infinie de la caméra.
while 1:
    #récupération des images de la caméra
    ret, img = cap.read()
    img=cv.flip(img,1)
    tempsActuel=time.time()
    if(tempsActuel-tempsDebut<3):
        overlay_image_alpha(img, liste_npGenerique[0][:, :, :3], (0, 0),
                            liste_npGenerique[0][:, :, 3] / 255.0)
    elif ((tempsActuel - tempsDebut > 3) & (tempsActuel - tempsDebut <=7)):
        overlay_image_alpha(img, liste_npGenerique[2][:, :, :3], (0, 0),
                            liste_npGenerique[2][:, :, 3] / 255.0)
    else:
        #liste contenant les émotion de tous les visages sur l'image, si tout le monde est joyeux: on prend la photo.
        listeEmotion=[]
        # find the face
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # On parcourt la liste des visages (elle contient sa position en x, y , sa largeur w et sa hauteur h)
        for index,(x,y,w,h) in enumerate(faces):
            #on repère les differents visages, fonction de leur position.
            visages, visageIndex = lissage(visages, visageIndex, frameNumber,x,y,w,h)

            #on crop l'image autour des visages.
            np_face = img[y:y+h, x:x+w]
            # converting into PIL.Image object to resize
            pil_face = Image.fromarray(np_face, 'RGB')
            pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)

            # remettre tout en np.array
            np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)
            superpose(img,np_face,0,49*index)
            #prediction
            predsMean=prediction(np_face,mean_image,std_image,visages,visageIndex,frameNumber)

            color = np.dot(predsMean[0], feelings_to_colors)
            color = color.astype(int).tolist()
            color_bgr = (color[2], color[1], color[0])
            cv.rectangle(img,(x,y), (x+w,y+h), color_bgr,2)
            listeEmotion.append(showFineResults(predsMean))
            color_rgb = tuple(color)

            img = write((x+5,y), img, showFineResults(predsMean), color_rgb, int(w/8))


        number,flash,lastpicture,time_tampon=photo(img,width,height,np_flash,listeEmotion,liste_npChiffre,lastpicture,time_tampon,number,flash)
    cv.imshow('img',img)


    # kill with ctr+c
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

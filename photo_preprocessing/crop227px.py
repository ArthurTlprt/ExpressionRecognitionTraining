import csv
import numpy as np
from PIL import Image, ImageDraw, ImageOps

happy=10000
sad=10000
neutral=10000
anger=10000
surprise=10000

with open('training.csv') as f:
    has_header = csv.Sniffer().has_header(f.read(1024))
    f.seek(0)  # rewind
    incsv = csv.reader(f)
    if has_header:
        next(incsv)  # skip header row
    reader = csv.reader(f)
    for row in reader:
        # opening image
        try:
            if(int(row[6])==0 & (neutral>0)):
                image = Image.open('../Manually_Annotated_Images/'+row[0]).convert('L')
                # croping face
                # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
                image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
                # resize
                # image = image.resize((64, 64), resample=Image.BILINEAR)
                image = image.resize((227, 227), resample=Image.BILINEAR)
                nomFichier=row[0].split('/')[1].split('.')[0]
                nomFichier=nomFichier+".pgm"
                image.save('../out227px/'+nomFichier)
                neutral=neutral-1
            elif(int(row[6])==1 & (happy>0)):
                image = Image.open('../Manually_Annotated_Images/'+row[0]).convert('L')
                # croping face
                # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
                image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
                # resize
                # image = image.resize((64, 64), resample=Image.BILINEAR)
                image = image.resize((227, 227), resample=Image.BILINEAR)
                nomFichier=row[0].split('/')[1].split('.')[0]
                nomFichier=nomFichier+".pgm"
                image.save('../out22px/'+nomFichier)
                happy=happy-1
            elif(int(row[6])==2 & (sad>0)):
                image = Image.open('../Manually_Annotated_Images/'+row[0]).convert('L')
                # croping face
                # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
                image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
                # resize
                # image = image.resize((64, 64), resample=Image.BILINEAR)
                image = image.resize((227, 227), resample=Image.BILINEAR)
                nomFichier=row[0].split('/')[1].split('.')[0]
                nomFichier=nomFichier+".pgm"
                image.save('../out227px/'+nomFichier)
                sad=sad-1
            elif(int(row[6])==3 & (surprise>0)):
                image = Image.open('../Manually_Annotated_Images/'+row[0]).convert('L')
                # croping face
                # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
                image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
                # resize
                # image = image.resize((64, 64), resample=Image.BILINEAR)
                image = image.resize((227, 227), resample=Image.BILINEAR)
                nomFichier=row[0].split('/')[1].split('.')[0]
                nomFichier=nomFichier+".pgm"
                image.save('../out227px/'+nomFichier)
                surprise=surprise-1
            elif(int(row[6])==6 & (anger>0)):
                image = Image.open('../Manually_Annotated_Images/'+row[0]).convert('L')
                # croping face
                # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
                image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
                # resize
                # image = image.resize((64, 64), resample=Image.BILINEAR)
                image = image.resize((227, 227), resample=Image.BILINEAR)
                nomFichier=row[0].split('/')[1].split('.')[0]
                nomFichier=nomFichier+".pgm"
                image.save('../out227px/'+nomFichier)
                anger=anger-1
        except IOError:
            print("image pas trouver")

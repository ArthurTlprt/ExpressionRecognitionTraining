import csv
import numpy as np
from PIL import Image, ImageDraw, ImageOps



with open('training.csv') as f:
    has_header = csv.Sniffer().has_header(f.read(1024))
    f.seek(0)  # rewind
    incsv = csv.reader(f)
    if has_header:
        next(incsv)  # skip header row
    reader = csv.reader(f)
    for row in reader:
        # opening image
        image = Image.open('../Manually_Annotated_Images/'+row[0])
        # croping face
        # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
        image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
        # resize
        image = image.resize((70, 70), resample=Image.BILINEAR)
        image.save('../out70px/'+row[0].split('/')[1])

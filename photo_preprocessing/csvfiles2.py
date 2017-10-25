import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio


data=[]
label_regression=[]
label_classification=[]
c=1
with open('training.csv') as f:
    has_header = csv.Sniffer().has_header(f.read(1024))
    f.seek(0)  # rewind
    incsv = csv.reader(f)
    if has_header:
        next(incsv)  # skip header row
    reader = csv.reader(f)
    for row in reader:
        print("cropping the image",c)
        #crop
         
        # opening image
        image = Image.open('Manually_Annotated_Images/'+row[0])
        # croping face
        # image = image.crop((Face_x, Face_y,Face_x+Face_width, Face_y+Face_height))
        image = image.crop((int(row[1]), int(row[2]),int(row[1])+int(row[3]), int(row[2])+int(row[4])))
        # resize
        image = image.resize((49, 49), resample=Image.BILINEAR)
        image.save('out/'+row[0].split('/')[1])
        # h5
        
        data.append(mio.import_image('out/'+row[0].split('/')[1]).pixels.astype(np.float32))
        # extracting usefull data
        label_regression.append([np.float32(row[7]), np.float32(row[8])])
        label_classification.append([np.uint8(row[6])])
        c+=1
        if c%30000 == 0:
            g = h5py.File("h5_2/training"+str(c/30000)+".hdf5", "w")
            g.create_dataset('label_regression', data=label_regression, dtype=np.float32)
            g.create_dataset('label_classification', data=label_classification, dtype=np.uint8)
            g.create_dataset('data', data=data,dtype=np.float32)
            g.close()
            data=[]
            label_regression=[]
            label_classification=[]
            
print("done")


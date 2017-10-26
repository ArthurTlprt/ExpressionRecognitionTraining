import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool

def load_image_thread(row):
    row = row.split(',')
    image = Image.open('../Manually_Annotated_Images/'+row[0])
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data

csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger', 'Non_Face']

for csv_name in csv_names:
    lines = [line.rstrip('\r\n') for line in open(csv_name)]
    if len(lines) == 60000:
        print("fichier de 60000 images, ok pour h5")

        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
        m.close()


        images = [x[0] for x in data]
        reg = [x[1] for x in data]
        classi = [x[2] for x in data]

        g = h5py.File("../classes/training"+csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()
    else:
        while len(lines) != 60000:
            

print("done")

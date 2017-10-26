import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool

def load_image_thread(row):
    row = row.split(',')
    image = Image.open('../Manually_Annotated_Images/'+row[0][1:-1])
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[7]), np.float32(row[8])]),
    np.uint8(row[6])
    ]
    g = h5py.File("../classes/training"+row[6]+".hdf5", "w")
    g.create_dataset('data', data=data[0],dtype=np.float32)
    g.create_dataset('label_regression', data=data[1],dtype=np.float32)
    g.create_dataset('label_classification', data=data[2],dtype=np.uint8)
    g.close()

csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Anger', 'Non_Face']

for csv_name in csv_names:
    lines = [line.rstrip('\r\n') for line in open(csv_name)]
    if len(lines) == 60000:
        print("fichier de 60000 images, ok pour h5")
        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
        m.close()

print("done")

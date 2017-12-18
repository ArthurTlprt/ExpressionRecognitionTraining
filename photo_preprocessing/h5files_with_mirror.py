import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial



def load_image_thread(row):
    row = row.split(',')
    try:
        image = Image.open('../Manually_Annotated_Images/'+row[0])
    except IOError:
        print("Image " + row[0] + "not found")
    #print("image access")
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data

def load_image_thread_mirror(row):

    row = row.split(',')
    try:
        image = Image.open('../Manually_Annotated_Images/'+row[0])
    except IOError:
        print("Image " + row[0] + "not found")
    #print("image access")
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.flipud(np.asarray(image).astype(np.float32)/255),
    np.array([np.float32(row[1]), np.float32(row[2])]),
    np.uint8(row[3])
    ]
    return data

csv_names = ['Sad','Anger','Surprise']

for csv_name in csv_names:
    print(csv_name)
    current_csv_read=open(csv_name,"r")
    lines = [line.rstrip('\r\n') for line in current_csv_read ]
    if len(lines) == 60000:
        print("fichier de 60000 images, ok pour h5")

        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
        print(np.shape(data))
        m.close()


        images = [x[0] for x in data]
        reg = [x[1] for x in data]
        classi = [x[2] for x in data]

        print("fin du chargement, création du fichier h5.")
        g = h5py.File("../h5_with_mirror/training"+csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()
    else:
        print("fichier <60000 images:")
        print("dataset sans flip...")
        m = ThreadPool(12)
        data = m.map(load_image_thread, lines)
        print(np.shape(data))
        m.close()


        images = [x[0] for x in data]
        reg = [x[1] for x in data]
        classi = [x[2] for x in data]

        print("avec flip...")
        m = ThreadPool(12)
        data = m.map(load_image_thread_mirror, lines)
        print(np.shape(data))
        m.close()


        images = images+[x[0] for x in data]
        reg = reg+[x[1] for x in data]
        classi = classi+ [x[2] for x in data]

        print("fin du chargement, création du fichier h5.")
        g = h5py.File("../h5_with_mirror/training"+csv_name+".hdf5", "w")
        g.create_dataset('data', data=images,dtype=np.float32)
        g.create_dataset('label_regression', data=reg,dtype=np.float32)
        g.create_dataset('label_classification', data=classi,dtype=np.uint8)
        g.close()










print("done")

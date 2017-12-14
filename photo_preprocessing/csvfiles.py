import csv
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import menpo.io as mio
from multiprocessing.dummy import Pool as ThreadPool


def load_image_thread(row):
    row = row.split(',')
    image = Image.open('Manually_Annotated_Images/'+row[0])
    #bound = int(int(row[3]) * 0.3)
    #image = image.crop((int(row[1]) - bound, int(row[2]) - bound,int(row[1])+int(row[3]) + (bound*2), int(row[2])+int(row[4])+ (bound*2)))
    #image.save('test'+str(row[0].split('/')[1]))
    image = image.resize((49, 49), resample=Image.BILINEAR)
    data = [ np.asarray(image).astype(np.float32)/255,
    np.array([np.float32(row[7]), np.float32(row[8])]),
    np.uint8(row[6])
    ]g = h5py.File("h5/training"+str(i+1)+".hdf5", "w")
    g.create_dataset('data', data=images,dtype=np.float32)
    g.create_dataset('label_regression', data=reg,dtype=np.float32)
    g.create_dataset('label_classification', data=classi,dtype=np.uint8)
    g.close()

    return data

data=[]

csv_data = [line.rstrip('\r\n') for line in open('training.csv')]
csv_data = csv_data[1:]

h5_size = 30000
for c,i in enumerate(range(7, 8)): #int(len(csv_data)/h5_size))):
    print(i)
    m = ThreadPool(12)
    data = m.map(load_image_thread, csv_data[(i*h5_size)+1:(i+1)*h5_size+1])
    m.close()

    images = [x[0] for x in data]
    reg = [x[1] for x in data]
    classi = [x[2] for x in data]

    g = h5py.File("h5/training"+str(i+1)+".hdf5", "w")
    g.create_dataset('data', data=images,dtype=np.float32)
    g.create_dataset('label_regression', data=reg,dtype=np.float32)
    g.create_dataset('label_classification', data=classi,dtype=np.uint8)
    g.close()

'''
reste=len(csv_data)-int(len(csv_data)/h5_size)*h5_size

m = ThreadPool(12)
data = m.map(load_image_thread, csv_data[int(len(csv_data)/h5_size)*h5_size:])
m.close()

images = [x[0] for x in data]
reg = [x[1] for x in data]
classi = [x[2] for x in data]

g = h5py.File("h5/validation0.hdf5", "w")
g.create_dataset('data', data=images,dtype=np.float32)
g.create_dataset('label_regression', data=reg,dtype=np.float32)
g.create_dataset('label_classification', data=classi,dtype=np.uint8)
g.close()'''
'''
for a in range(reste):
    g = h5py.File("h5/training0.hdf5", "w")
    g.create_dataset('data', data=images[int(len(csv_data)/h5_size)*h5_size+a+1],dtype=np.float32)
    g.create_dataset('label_regression', data=reg[int(len(csv_data)/h5_size)*h5_size+a+1],dtype=np.float32)
    g.create_dataset('label_classification', data=classi[int(len(csv_data)/h5_size)*h5_size+a+1],dtype=np.uint8)
    g.close()
    print(c)
  '''

print("done")

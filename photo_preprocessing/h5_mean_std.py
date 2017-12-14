import sys
import csv
import os
import glob
import h5py
import cv2
import numpy as np
import menpo.image as mi
import menpo.io as mio

h5 = []
names = ['Anger','Neutral', 'Happy', 'Sad', 'Surprise']


for name in names:
    f = h5py.File('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/ShuffledBW/training' + name + '.hdf5', 'r')
    h5.append(np.copy(f['data'][:28000]))
    f.close()

mean_image = np.zeros((64,64), dtype=np.float32)
std_image = np.zeros((64,64), dtype=np.float32)


db_size = 5*28000
print(db_size)

print('Compute mean image...')
for images in h5:
    for image in images:

        #mean_image = mean_image + image.transpose(2, 0, 1)/db_size
        mean_image = mean_image + image/db_size
mio.export_image(mi.Image(mean_image), "/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/mean_image.png", overwrite=True)


print('Compute std image...')
for images in h5:
    for image in images:
        std_image = std_image + np.power((image-mean_image), 2)/db_size
std_image = np.sqrt(std_image)
mio.export_image(mi.Image(std_image), "/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/std_image.png", overwrite=True)


print('saving in h5 file')
g = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/mean_std.hdf5", "w")
g.create_dataset('mean', data=mean_image,dtype=np.float32)
g.create_dataset('std', data=std_image,dtype=np.float32)

print('Done.')

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
names = ['Anger','Neutral', 'Happy', 'Sad', 'Surprise', 'Non_Face']


for name in names:
    f = h5py.File('../classes/training' + name + '.hdf5', 'r')
    h5.append(np.copy(f['data'][:60000]))
    f.close()

mean_image = np.zeros((3, 49, 49), dtype=np.float32)
std_image = np.zeros((3, 49, 49), dtype=np.float32)


db_size = 360000
print(db_size)

print('Compute mean image...')
for images in h5:
    for image in images:
        print(image.shape)
        mean_image = mean_image + image.transpose(2, 0, 1)/db_size
mio.export_image(mi.Image(mean_image), "../mean_std/mean_image.png", overwrite=True)


print('Compute std image...')
for images in h5:
    for image in images:
        std_image = std_image + np.power((image.transpose(2, 0, 1)-mean_image), 2)/db_size
std_image = np.sqrt(std_image)
mio.export_image(mi.Image(std_image), "../mean_std/std_image.png", overwrite=True)


print('saving in h5 file')
g = h5py.File("../mean_std/mean_std.hdf5", "w")
g.create_dataset('mean', data=mean_image,dtype=np.float32)
g.create_dataset('std', data=std_image,dtype=np.float32)

print('Done.')

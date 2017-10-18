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

h5_files = [line.rstrip('\r\n') for line in open('h5_files.txt')]
for h5_file in h5_files:
    f = h5py.File('h5/' + h5_file, 'r')
    h5.append(np.copy(f['data']))

    f.close()

mean_image = np.zeros((3, 49, 49), dtype=np.float32)
std_image = np.zeros((3, 49, 49), dtype=np.float32)


indexlist = [line.rstrip('\r\n') for line in open('training.csv')]
db_size = len(indexlist)
print(db_size)

print('Compute mean image...')
for images in h5:
    for image in images:
        mean_image = mean_image + np.transpose(image)/db_size
mio.export_image(mi.Image(mean_image), "mean_std/mean_image.png", overwrite=True)


print('Compute std image...')
for images in h5:
    for image in images:
        std_image = std_image + np.power((np.transpose(image)-mean_image), 2)/db_size
std_image = np.sqrt(std_image)
mio.export_image(mi.Image(std_image), "mean_std/std_image.png", overwrite=True)


print('saving in h5 file')
g = h5py.File("mean_std/mean_std.hdf5", "w")
g.create_dataset('mean', data=mean_image,dtype=np.float32)
g.create_dataset('std', data=std_image,dtype=np.float32)

print('Done.')

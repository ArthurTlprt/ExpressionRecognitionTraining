import os
import time
import h5py
import numpy as np
import random

# label_regression
# label_classification
# data
def swap(liste_index, liste_donne):
    list_return=[]
    for i in range(10000):
        list_return.append(liste_donne[liste_index[i]])
    return list_return

csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']
liste_index=[]
for i in range(10000):
    liste_index.append(i)
random.shuffle(liste_index)

for i, csv_name in enumerate(csv_names):
    f = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classes227/training"+csv_name+".hdf5", 'r+')
    print(f)
    liste_regression=f['label_regression']
    liste_classes=f['label_classification']
    liste_data=f['data']
    print(liste_regression[0][0])
    liste_regression=swap(liste_index, liste_regression)
    liste_classes=swap(liste_index, liste_classes)
    liste_data=swap(liste_index,liste_data)
    print(liste_regression[0][0])
    g = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classes227/Shuffled227/training"+csv_name+".hdf5", "w")
    g.create_dataset('data', data=liste_data,dtype=np.float32)
    g.create_dataset('label_regression', data=liste_regression,dtype=np.float32)
    g.create_dataset('label_classification', data=liste_classes,dtype=np.uint8)
    g.close()
    # f['label_regression'] = liste_regression
    # f['label_classification'] = liste_classes
    # f['data'] = liste_data
    f.close()

import numpy as np
import h5py
from random import shuffle as S
import keras
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, BatchNormalization, Activation, Lambda, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger






model=load_model('inception_resnet.h5')
img_path = 'contente.png'
img = image.load_img(img_path, target_size=(49,49))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


preds= model.predict(x)


print(preds)

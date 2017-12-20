import numpy as np
import h5py
from random import shuffle as S
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Add, AveragePooling2D, GlobalAveragePooling2D
from PIL import Image, ImageDraw, ImageOps
import keras.callbacks as C

#import tensorflow as tf


# Callback

class LossHistory(C.Callback):

    def on_train_begin(self, logs={}):
        #initialization of the empty dataset
        self.losses = []

        self.weights = {}
        for layer in model.layers:
            if type(layer) is Dense:
                self.weights[layer.name] = []

    def on_epoch_end(self, epoch, logs={}):
        # saving dataset into csv
        losses_to_save = np.array(self.losses, dtype='float32')
        np.savetxt('loss.csv', losses_to_save, header="x,y", comments="", delimiter=",")

        for dense in self.weights:
            weights_to_save = np.array(self.weights[dense], dtype='float32')
            filename = dense +'.csv'
            np.savetxt(filename, weights_to_save, header="x,y", comments="", delimiter=",")

    def on_batch_end(self, batch, logs={}):
        # at each batch we compute historic
        self.losses.append([len(self.losses), logs.get('loss')])
        for l in model.layers:
            if type(l) is Dense:
                weights = l.get_weights()[0].flatten()
                bias = l.get_weights()[1].flatten()
                self.weights[l.name].append([len(self.losses), self.mean_magnitude(weights, bias)])

    def mean_magnitude(self, weights, bias):
        mean_magnitude = np.append(weights, bias)
        n = float(mean_magnitude.size)
        mean_magnitude = np.square(mean_magnitude)
        mean_magnitude = np.sum(mean_magnitude)
        mean_magnitude = np.divide(mean_magnitude, n)
        mean_magnitude = np.sqrt(mean_magnitude)
        mean_magnitude = np.log10(mean_magnitude)
        return mean_magnitude


def generate_arrays_from_file(images, labels, batch_size=16, shuffle=True):
    x = np.zeros((batch_size, 227, 227,1))
    y = np.zeros((batch_size, 5))
    batch_id = 0
    idx = np.arange(0, len(images))
    if shuffle:
        #c = list(zip(images, labels))
        S(idx)
        #images, labels = zip(*c)
    while 1:
        for i in idx:
            x[batch_id, ...] = images[i].reshape(227,227,1)
            y[batch_id, ...] = keras.utils.to_categorical(labels[i], num_classes=5)
            if batch_id == (batch_size - 1):
                yield (x, y)
                batch_id = 0
            else:
                batch_id += 1

def custom_conv2d(x, filters, kernel, strides, padding, normalize, activation):

    x = Conv2D(filters, kernel, strides=strides, padding=padding)(x)

    if normalize == True:
        x = BatchNormalization()(x)

    if activation != None:
        x = Activation(activation)(x)

    return x


def architecture():
    i = Input(shape=(227, 227, 1))
    print(i,type(i))

    print("C1")
    x = custom_conv2d(i, 96, (11, 11), (4, 4), 'valid', True, 'relu')
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(x)

    print("C2")
    x = custom_conv2d(x, 128, (5, 5), (1, 1), 'same', True, 'relu')
    print(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    print("C3")
    x = custom_conv2d(x, 192, (3, 3), (1, 1), 'same', True, 'relu')
    print(x)

    print("C4")
    x = custom_conv2d(x, 192, (3, 3), (1, 1), 'same', True, 'relu')
    print(x)

    print("C5")
    x = custom_conv2d(x, 128, (3, 3), (1, 1), 'same', True, 'relu')
    print(x,type(x))

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    print("Dense 1")
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    #x = tf.reshape(x, [-1, 1024])
    print(x,type(x))

    print("Dense 2")
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    print(x)

    print("Dense 3")

    o = Dense(5, activation='softmax')(x)
    print(o,type(o))
    model = Model(inputs=i, outputs=o)

    model.summary()

    return model


def load_mean_std(h5_path):

    f = h5py.File(h5_path, 'r')
    mean_image = np.copy(f['mean'])
    std_image = np.copy(f['std'])
    f.close()

    return mean_image, std_image

def load_data():


    csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']

    images_training = np.zeros((5*train_size, 227, 227), np.float32)
    images_validation = np.zeros((5*val_size, 227, 227), np.float32)
    annotations_training = np.zeros((5*train_size), np.uint8)
    annotations_validation = np.zeros((5*val_size), np.uint8)

    for i, csv_name in enumerate(csv_names):
        print(i)
        f = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classes227/Shuffled227/training"+csv_name+".hdf5", 'r')
        images_training[i*train_size:(i+1)*train_size] = f['data'][:train_size]
        images_validation[i*val_size:(i+1)*val_size] = f['data'][train_size:class_size]
        annotations_training[i*train_size:(i+1)*train_size] = np.full((train_size), i, dtype=np.uint8)
        annotations_validation[i*val_size:(i+1)*val_size] = np.full((val_size), i, dtype=np.uint8)
        f.close()

    return images_training, annotations_training, images_validation, annotations_validation


def normalize_image(image):
    return np.divide((image - mean_image), std_image)

train_size = 8000
val_size = 2000
class_size = 10000
batch_size = 16
mean_image, std_image = load_mean_std('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classes227/mean_std.hdf5')


if __name__ == "__main__":

    print('Loading dataset...')

    images_training, annotations_training, images_validation, annotations_validation = load_data()


    for i, image in enumerate(images_training):
        images_training[i] = normalize_image(image)

    for i, image in enumerate(images_validation):
        images_validation[i] = normalize_image(image)

    print('Network...')
    model = architecture()

    print('Optimization...')
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print('Training...')
    checkpointer = ModelCheckpoint(filepath='snapshotsBW/irc-cnn-{epoch:03d}-{val_loss:.6f}.h5', verbose=1, save_best_only=True)
    hist = model.fit_generator(generate_arrays_from_file(images_training, annotations_training), int(train_size*5/batch_size), epochs=500, callbacks=[LossHistory(),checkpointer], validation_data=generate_arrays_from_file(images_validation, annotations_validation, shuffle=False), validation_steps=int(val_size*5/batch_size), max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)

    print('Recording...')
    model.save('8layers.h5')

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
import tensorflow as tf



def generate_arrays_from_file(images, labels, batch_size=32, shuffle=True):
    x = np.zeros((batch_size, 64, 64,1))
    y = np.zeros((batch_size, 5))
    batch_id = 0
    idx = np.arange(0, len(images))
    if shuffle:
        #c = list(zip(images, labels))
        S(idx)
        #images, labels = zip(*c)
    while 1:
        for i in idx:
            x[batch_id, ...] = images[i].reshape(64,64,1)
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

    print("C1")
    x = custom_conv2d(i, 96, (11, 11), (1, 1), 'valid', True, 'relu')
    print(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(x)
    print(x)

    print("C2")
    x = custom_conv2d(x, 128, (5, 5), (1, 1), 'valid', True, 'relu')
    print(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(x)

    print("C3")
    x = custom_conv2d(x, 192, (3, 3), (2, 2), 'valid', True, 'relu')
    print(x)

    print("C4")
    x = custom_conv2d(x, 192, (3, 3), (2, 2), 'valid', True, 'relu')
    print(x)

    print("C5")
    x = custom_conv2d(x, 128, (3, 3), (2, 2), 'valid', True, 'relu')
    print(x)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    o = Dense(5, activation='softmax')(x)

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

    images_training = np.zeros((5*48000, 64, 64), np.float32)
    images_validation = np.zeros((5*12000, 64, 64), np.float32)
    annotations_training = np.zeros((5*48000), np.uint8)
    annotations_validation = np.zeros((5*12000), np.uint8)

    for i, csv_name in enumerate(csv_names):
        print(i)
        f = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/ShuffledBW/training"+csv_name+".hdf5", 'r')
        images_training[i*48000:(i+1)*48000] = f['data'][:48000]
        images_validation[i*12000:(i+1)*12000] = f['data'][48000:60000]
        annotations_training[i*48000:(i+1)*48000] = np.full((48000), i, dtype=np.uint8)
        annotations_validation[i*12000:(i+1)*12000] = np.full((12000), i, dtype=np.uint8)
        f.close()

    return images_training, annotations_training, images_validation, annotations_validation


def normalize_image(image):
    return np.divide((image - mean_image), std_image)


mean_image, std_image = load_mean_std('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/classesBW/mean_std.hdf5')


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
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs_v2', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    csv_logger = CSVLogger('irc-cnn.log', append=True)
    checkpointer = ModelCheckpoint(filepath='snapshotsBW/irc-cnn-{epoch:03d}-{val_loss:.6f}.h5', verbose=1, save_best_only=True)
    hist = model.fit_generator(generate_arrays_from_file(images_training, annotations_training), int(48000*5/32), epochs=500, callbacks=[csv_logger, checkpointer], validation_data=generate_arrays_from_file(images_validation, annotations_validation, shuffle=False), validation_steps=int(12000*5/32), max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)

    print('Recording...')
    model.save('8layers.h5')

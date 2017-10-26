import numpy as np
import h5py
from random import shuffle as S
import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Add, AveragePooling2D

def generate_arrays_from_file(images, labels, shuffle=True, batch_size=32):
    x = np.zeros((batch_size, 49, 49, 3))
    y = np.zeros((batch_size, 11))
    batch_id = 0
    while 1:
        ids_list = np.arange(0, len(images))
        if shuffle:
            S(ids_list)
        for id_list in ids_list:
            #ids_im = np.arange(0, len(images[id_list]))
            #if shuffle:
            #    S(ids_im)
            for i in range(0, len(images[id_list])):
                x[batch_id, ...] = images[id_list][i]
                y[batch_id, ...] = keras.utils.to_categorical(labels[id_list][i], num_classes=11)
                if batch_id == (batch_size - 1):            
                    yield (x, y)
                    batch_id = 0
                else:
                    batch_id += 1

def custom_conv2d(x, filters, kernel, strides, padding, normalize, activation):

    x = Conv2D(filters, kernel, strides=strides, padding=padding)(x)

    if (normalize==True):
	    x = BatchNormalization()(x)

    if (activation!=None):
        x = Activation(activation)(x)

    return x

def inception_resnet_v2_stem(x):

    x = custom_conv2d(x, 32, (3, 3), (2, 2), 'valid', True, 'relu')
    x = custom_conv2d(x, 32, (3, 3), (1, 1), 'same', True, 'relu')
    x = custom_conv2d(x, 64, (3, 3), (1, 1), 'same', True, 'relu')

    x_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x_b = custom_conv2d(x, 96, (3, 3), (2, 2), 'valid', True, 'relu')

    x = Concatenate(axis=-1)([x_a, x_b])

    x_a = custom_conv2d(x, 64, (1, 1), (1, 1), 'same', True, 'relu')
    x_a = custom_conv2d(x_a, 96, (3, 3), (1, 1), 'valid', True, 'relu')
    x_b = custom_conv2d(x, 64, (1, 1), (1, 1), 'same', True, 'relu')
    x_b = custom_conv2d(x_b, 64, (7, 1), (1, 1), 'same', True, 'relu')
    x_b = custom_conv2d(x_b, 64, (1, 7), (1, 1), 'same', True, 'relu')
    x_b = custom_conv2d(x_b, 96, (3, 3), (1, 1), 'valid', True, 'relu')

    x = Concatenate(axis=-1)([x_a, x_b])

    x_a = custom_conv2d(x, 192, (3, 3), (1, 1), 'same', True, 'relu')
    x_b = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)

    x = Concatenate(axis=-1)([x_a, x_b])

    return x

def inception_resnet_v2_a(x, scale=0.17):

	x_a = Activation('relu')(x)

	x_b_1 = custom_conv2d(x, 32, (1, 1), (1, 1), 'same', True, 'relu')

	x_b_2 = custom_conv2d(x, 32, (1, 1), (1, 1), 'same', True, 'relu')
	x_b_2 = custom_conv2d(x_b_2, 32, (3, 3), (1, 1), 'same', True, 'relu')

	x_b_3 = custom_conv2d(x, 32, (1, 1), (1, 1), 'same', True, 'relu')
	x_b_3 = custom_conv2d(x_b_3, 48, (3, 3), (1, 1), 'same', True, 'relu')
	x_b_3 = custom_conv2d(x_b_3, 64, (3, 3), (1, 1), 'same', True, 'relu')

	x_b = Concatenate(axis=-1)([x_b_1, x_b_2, x_b_3])

	x_b = custom_conv2d(x, 384, (1, 1), (1, 1), 'same', True, None)

	# x_a = Lambda(lambda x: x * scale)(x_a)

	x = Add()([x_a, x_b])

	x = Activation('relu')(x)

	return x

def inception_resnet_v2():

    i = Input(shape=(49, 49, 3))

    x = inception_resnet_v2_stem(i)
    x = inception_resnet_v2_a(x)

    #x = AveragePooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(11)(x)
    o = Activation('softmax')(x)

    model = Model(inputs=i, outputs=o)

    model.summary()

    return model

def load_mean_std(h5_path):

    f = h5py.File(h5_path, 'r')
    mean_image = np.copy(f['mean']).transpose(1, 2, 0)
    std_image = np.copy(f['std']).transpose(1, 2, 0)
    f.close()

    return mean_image, std_image

def load_data(txt_path):

    images = []
    annotations = []
    h5_files = [line.rstrip('\r\n') for line in open(txt_path)]
    for h5_file in h5_files:
        f = h5py.File("/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/h5/" + h5_file, 'r')
        images.append(np.copy(f['data']))
        annotations.append(np.copy(f['label_classification']))
        f.close()

    return images, annotations

def normalize_image(image):
    return np.divide((image - mean_image), std_image)

if __name__ == "__main__":

    print('Loading dataset...')

    mean_image, std_image = load_mean_std('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/mean_std/mean_std.hdf5')

    images_training, annotations_training = load_data('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/h5_training.txt')

    images_validation, annotations_validation = load_data('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/h5_validation.txt')

    for eit, images in enumerate(images_training):
        for ei, image in enumerate(images):
            images_training[eit][ei] = normalize_image(image)

    for eit, images in enumerate(images_validation):
        for ei, image in enumerate(images):
            images_validation[eit][ei] = normalize_image(image)


    print('Network...')
    model = inception_resnet_v2()

    print('Optimization...')
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    #plot_model(model, to_file='model.png')

    indexlist = [line.rstrip('\r\n') for line in open('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/training.csv')]
    db_size = len(indexlist)

    print('Training...')

    csv_logger = CSVLogger('irc-cnn.log', append=True)
    checkpointer = ModelCheckpoint(filepath='snapshots/irc-cnn-{epoch:03d}-{val_loss:.6f}.h5', verbose=1, save_best_only=True)
    hist = model.fit_generator(generate_arrays_from_file(images_training, annotations_training), int(360000/32)-1, epochs=500, callbacks=[csv_logger, checkpointer], validation_data=generate_arrays_from_file(images_validation, annotations_validation), validation_steps=int(5500/32)-1, max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)

    print('Recording...')
    model.save('irc-cnn.h5')

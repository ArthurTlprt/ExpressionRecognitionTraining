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



def generate_arrays_from_file(images, labels, batch_size=32, shuffle=True):
    x = np.zeros((batch_size, 49, 49, 3))
    y = np.zeros((batch_size, 5))
    batch_id = 0
    if shuffle:
        c = list(zip(images, labels))
        S(c)
        images, labels = zip(*c)
    while 1:
        for i in range(0, len(images)):
            x[batch_id, ...] = images[i]
            y[batch_id, ...] = keras.utils.to_categorical(labels[i], num_classes=5)
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
    x = Dense(5)(x)
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

def load_data():

    f = h5py.File("../classes/trainingNeutral.hdf5", 'r')

    # images_training = f['data'][:48000]
    # images_validation = f['data'][48000:60000]
    # annotations_training = f['label_classification'][:48000]
    # annotations_validation = f['label_classification'][48000:60000]
    # f.close()
    #
    # # csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger', 'Non_Face']
    # csv_names = ['Neutral', 'Happy', 'Sad']
    # for csv_name in csv_names[1:]:
    #     f = h5py.File("../classes/training"+csv_name+".hdf5", 'r')
    #     print("loading...")
    #     images_training = np.concatenate([images_training, f['data'][:48000]])
    #     images_validation = np.concatenate([images_validation, f['data'][48000:60000]])
    #     annotations_training = np.concatenate([annotations_training, f['label_classification'][:48000]])
    #     annotations_validation = np.concatenate([annotations_validation, f['label_classification'][48000:60000]])
    #     print(images_training.shape)
    #     f.close()

    csv_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']

    images_training = np.zeros((6*48000, 49, 49, 3), np.float32)
    images_validation = np.zeros((6*12000, 49, 49, 3), np.float32)
    annotations_training = np.zeros((6*48000), np.uint8)
    annotations_validation = np.zeros((6*12000), np.uint8)

    for i, csv_name in enumerate(csv_names):
        print(i)
        f = h5py.File("../classes/training"+csv_name+".hdf5", 'r')
        images_training[i*48000:(i+1)*48000] = f['data'][:48000]
        images_validation[i*12000:(i+1)*12000] = f['data'][48000:60000]
        annotations_training[i*48000:(i+1)*48000] = np.full((48000), i, dtype=np.uint8)
        annotations_validation[i*12000:(i+1)*12000] = np.full((12000), i, dtype=np.uint8)
        f.close()

    # index = np.arange(images_training.shape[0])
    # np.random.shuffle(index)
    # return images_training[index], annotations_training[index], images_validation, annotations_validation
    return images_training, annotations_training, images_validation, annotations_validation

def normalize_image(image):
    return np.divide((image - mean_image), std_image)


mean_image, std_image = load_mean_std('/media/isen/Data_windows/PROJET_M1_DL/Affect-Net/MAN/mean_std/mean_std.hdf5')


if __name__ == "__main__":

    print('Loading dataset...')

    images_training, annotations_training, images_validation, annotations_validation = load_data()


    for i, image in enumerate(images_training):
        images_training[i] = normalize_image(image)

    for i, image in enumerate(images_validation):
        images_validation[i] = normalize_image(image)

    print('Network...')
    model = inception_resnet_v2()

    print('Optimization...')
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])


    print('Training...')
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    csv_logger = CSVLogger('irc-cnn.log', append=True)
    checkpointer = ModelCheckpoint(filepath='snapshots/irc-cnn-{epoch:03d}-{val_loss:.6f}.h5', verbose=1, save_best_only=True)
    hist = model.fit_generator(generate_arrays_from_file(images_training, annotations_training), int(48000*5/32), epochs=500, callbacks=[csv_logger, checkpointer], validation_data=generate_arrays_from_file(images_validation, annotations_validation, shuffle=False), validation_steps=int(12000*5/32), max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)

    print('Recording...')
    model.save('irc-cnn.h5')

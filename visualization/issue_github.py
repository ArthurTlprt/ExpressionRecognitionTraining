from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.callbacks as C
import numpy as np
from math import log10
import tensorflow as tf
import keras.backend as K

model = Sequential()
model.add(Dense(10, input_dim=1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')


x_train = np.random.random((1000))
y_train = np.sin(x_train)

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0)


weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

print (weights)
# ==> [dense_1_W, dense_1_b]

import keras.backend as K

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)


from keras.utils.np_utils import to_categorical

inputs = [[[1, 2]], # X
          [1], # sample weights
          [[1]], # y
          0 # learning phase in TEST mode
]

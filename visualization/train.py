from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.callbacks as C
import numpy as np
from math import log10
import tensorflow as tf
import keras.backend as K

class LossHistory(C.Callback):

    def on_train_begin(self, logs={}):
        #initialization of the empty dataset
        self.losses = []
        self.weights = []
        self.gradients = []
        self.gradientsList = []

        print(len(model.layers))
        for l in model.layers:
            if type(l) is Dense:
                self.weights.append([])

    def on_train_end(self, logs={}):
        # saving dataset into csv

        losses_to_save = np.array(self.losses, dtype='float32')
        np.savetxt('loss.csv', losses_to_save, header="x,y", comments="", delimiter=",")

        for i, w in enumerate(self.weights):
            weights_to_save = np.array(w, dtype='float32')
            filename = 'weights'+ str(i)+'.csv'
            np.savetxt(filename, weights_to_save, header="x,y", comments="", delimiter=",")

    def on_batch_end(self, batch, logs={}):
        # at each batch we compute historic
        self.losses.append([len(self.losses), logs.get('loss')])
        for i,l in enumerate(model.layers):
            if type(l) is Dense:
                weights = l.get_weights()[0]
                # print(model.total_loss)
                # print(weights)
                # print(K.gradients(model.total_loss, weights))
                bias = l.get_weights()[1]
                self.weights[int(i/2)].append([len(self.losses), self.mean_magnitude(weights, bias)])
        # getting gradients
        # input_tensors=[model.inputs[0],model.sample_weights[0],model.targets[0],K.learning_phase(),]
        # weights = model.trainable_weights # weight tensors
        # weights = [weight for weight in weights if model.get_layer(weight.name.split('/')[0]).trainable] # filter down weights tensors to only ones which are trainable
        # gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
        # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
        # inputs = [x_train, # X
        #   [1], # sample weights
        #   y_train, # y
        #   0 # learning phase in TEST mode
        # ]
        # for i,l in enumerate(model.layers): #get weights into arrays instead of tf variable
        # 	if type(l) is Dense:
        # 		weights = l.get_weights()[0]
        # 		bias = l.get_weights()[1]
        #
        # self.gradientsList.append([a for a in zip(weights, get_gradients(inputs))])
        # print(self.gradientsList)


    def mean_magnitude(self, weights, bias):
        mean_magnitude = np.append(weights, bias)
        mean_magnitude = [w**2 for w in weights]
        mean_magnitude = sum(sum(mean_magnitude))
        mean_magnitude /= len(weights)
        mean_magnitude = mean_magnitude**(0.5)
        return log10(mean_magnitude)



model = Sequential()
model.add(Dense(10, input_dim=1, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')


x_train = np.random.random((1000, 1))
y_train = np.sin(x_train)
print(x_train[0])
print(y_train[0])

model.fit(x_train, y_train, batch_size=128, epochs=1000, verbose=0, callbacks=[LossHistory()])

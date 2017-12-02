from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.callbacks as C
import numpy as np

class LossHistory(C.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.gradients = []

    def on_epoch_end(self, epoch, logs={}):

        self.losses.append([epoch, logs.get('loss')])
        losses_to_save = np.array(self.losses, dtype='float32')
        np.savetxt('loss.csv', losses_to_save, header="x,y", comments="", delimiter=",")

        for l in model.layers:
            if type(l) is Dense:
                print('couche')
                weights = l.get_weights()[0]
                bias = l.get_weights()[1]
                print(self.mean_magnitude(weights, bias))

    def mean_magnitude(self, weights, bias):
        
        mean_magnitude = np.append(weights, bias)
        mean_magnitude = [w**2 for w in weights]
        mean_magnitude = sum(sum(mean_magnitude))
        mean_magnitude /= len(weights)
        mean_magnitude = mean_magnitude**(0.5)
        return mean_magnitude


model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
# model.add(Dense(10))
# model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


history = LossHistory()


x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

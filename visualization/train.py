from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.callbacks as C
import numpy as np

class LossHistory(C.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append([epoch, logs.get('loss')])
        losses_to_save = np.array(self.losses, dtype='float32')
        print(losses_to_save)
        np.savetxt('loss.csv', losses_to_save, header="x,y", comments="")

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()


x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)

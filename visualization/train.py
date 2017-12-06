from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.callbacks as C
import numpy as np
from math import log10

class LossHistory(C.Callback):

    def on_train_begin(self, logs={}):
        #initialization of the empty dataset
        self.losses = []
        self.weights = []
        self.gradients = []
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
                print('couche ', i)
                weights = l.get_weights()[0]
                bias = l.get_weights()[1]
                self.weights[int(i/2)].append([len(self.losses), self.mean_magnitude(weights, bias)])


    def mean_magnitude(self, weights, bias):
        mean_magnitude = np.append(weights, bias)
        mean_magnitude = [w**2 for w in weights]
        mean_magnitude = sum(sum(mean_magnitude))
        mean_magnitude /= len(weights)
        mean_magnitude = mean_magnitude**(0.5)
        return log10(mean_magnitude)


model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# gradient https://stackoverflow.com/questions/39561560/getting-gradient-of-model-output-w-r-t-weights-using-keras

# from keras import backend as k
# outputTensor = model.output
# listOfVariableTensors = model.trainable_weights
# gradients = k.gradients(outputTensor, listOfVariableTensors)
#
# import tensorflow as tf
# trainingExample = np.random.random((1,784))
# sess = tf.InteractiveSession()
# sess.run(tf.initialize_all_variables())
# evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
# print(evaluated_gradients)


history = LossHistory()


x_train = np.random.random((1000, 784))
y_train = np.random.random((1000, 10))

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Input
import keras.callbacks as C
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.utils.np_utils import to_categorical
import tensorflow as tf



class gradientHistory(C.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.gradientsList=[]
		self.weights = {}
		for layer in model.layers:
			if type(layer) is Dense:
				self.weights[layer.name] = []
		self.grads = {}
		for layer in model.layers:
			if type(layer) is Dense:
				self.grads[layer.name] = []
	def on_train_end(self, logs={}):
		# saving dataset into csv
		losses_to_save = np.array(self.losses, dtype='float32')
		np.savetxt('loss.csv', losses_to_save, header="x,y", comments="", delimiter=",")
		for dense in self.weights:
			weights_to_save = np.array(self.weights[dense], dtype='float32')
			filename = dense +'_weights'+'.csv'
			np.savetxt(filename, weights_to_save, header="x,y", comments="", delimiter=",")
		for dense in self.grads:
			grads_to_save= np.array(self.grads[dense], dtype='float32')
			filename = dense +'_grads'+'.csv'
			np.savetxt(filename, grads_to_save, header="x,y", comments="", delimiter=",")



	def on_epoch_end(self, epoch, logs={}):
		input_tensors=[model.inputs[0],model.sample_weights[0],model.targets[0],K.learning_phase(),]
		weights = model.trainable_weights # weight tensors
		weights = [weight for weight in weights if model.get_layer(weight.name.split('/')[0]).trainable] # filter down weights tensors to only ones which are trainable
		gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
		get_gradients = K.function(inputs=input_tensors, outputs=gradients)
		inputs = [x_train, # X
          [1], # sample weights
          y_train, # y
          0 # learning phase in TEST mode
		]
		self.gradientsList=(get_gradients(inputs))
		for i,l in enumerate(model.layers): #get weights into arrays instead of tf variable
			if type(l) is Dense:
				weights = l.get_weights()[0]#.flatten()
				bias = l.get_weights()[1]#.flatten()
				print("weights:")
				self.weights[l.name].append([len(self.losses), self.mean_magnitude(np.append(weights.flatten(), bias.flatten()))])
				print("gradients")
				self.grads[l.name].append([len(self.losses), self.mean_magnitude(self.gradientsList[i].flatten())])
		self.losses.append([len(self.losses), logs.get('loss')])
	def mean_magnitude(self, data):
		mean_magnitude = data
		n = float(mean_magnitude.size)
		mean_magnitude = np.square(mean_magnitude)
		mean_magnitude = np.sum(mean_magnitude)
		mean_magnitude = np.divide(mean_magnitude, n)
		mean_magnitude = np.sqrt(mean_magnitude)
		mean_magnitude = np.log10(mean_magnitude)
		print(mean_magnitude)
		return mean_magnitude


model = Sequential()
model.add(Dense(10, input_dim=3	, kernel_initializer='uniform'))
#model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#history = LossHistory()

x_train = np.random.random((1000, 3))
y_train = np.random.random((1000, 10))

#for t in model.trainable_weights:
#		print(t.name)

#print(tf.gradients(x_train,y_train))
history = gradientHistory()
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,callbacks=[history])

#input_tensors = [model.inputs[0], # input data
#                 model.sample_weights[0], # how much to weight each sample by
#                 model.targets[0], # labels
#                 K.learning_phase(), # train or test mode
#]
#
#
#
#
#weights = model.trainable_weights # weight tensors
#weights = [weight for weight in weights if model.get_layer(weight.name.split('/')[0]).trainable] # filter down weights tensors to only ones which are trainable
#gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors
#
#print(weights)
#
#
#get_gradients = K.function(inputs=input_tensors, outputs=gradients)
#
#inputs = [x_train, # X
#          [1], # sample weights
#          y_train, # y
#          0 # learning phase in TEST mode
#]
#
#print( [a for a in zip(weights, get_gradients(inputs))])
## ==> [(dense_1_W, array([[-0.42342907],
#
##print(get_gradients(model))

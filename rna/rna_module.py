import talos as ta
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import keras.preprocessing.text
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import EarlyStopping


class RnaModule(object):
	#conjuto de exemplos de treino
	data_set_samples = []
	#classes dos exemplos de treino
	data_set_labels = []
	#conjunto de exemplos de teste
	test_data_set_samples = []
	#classes dos exemplos de teste
	test_data_set_labels = []

	number_neurons_input_layer = 0
	number_neurons_hidden_layer = 0
	number_neurons_output_layer = 0

	input_dim_neurons = 0

	#funcoes de ativacao dos neuronios de cada camada
	activation_function_input_layer = "relu"
	activation_function_hidden_layer = "relu"
	activation_function_output_layer = "sigmoid"

	model = None
	hyperparameters_setting_mode = None

	def __init__(self):
		print("init rna module")

	def __generateModelHO(self,x_train, y_train, x_val, y_val, params):
		self.model = Sequential()
		self.model.add(Dense(params['n_neurons_input'], input_dim= self.input_dim_neurons, init='normal', activation=params['activation_function']))
		self.model.add(Dense(params['n_neurons_hidden'], init='normal', activation=params['activation_function']))
		self.model.add(Dense(1, init='normal', activation=params['activation_function']))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		csv_logger = CSVLogger('training.log')

		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss',patience=20)
		fit = self.model.fit(x_train, y_train, epochs=500, verbose=2, callbacks=[early_stopping])

		return fit, self.model
	#Faz a otimização de hiperparâmetros
	def doHO(self, x, y):
		p = {
			'n_neurons_input': [10, 20, 41],
			'n_neurons_hidden': [10, 20, 41],
			'activation_function': ['sigmoid', 'relu', 'elu'],
		}
		self.hyperparameters_setting_mode = "auto"
		print(talos.scan(x, y, p, self.__generateModelHO))

	#funcao para criar a rna para abordagem simples
	def generateModel(self):
		self.model = Sequential()
		self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		csv_logger = CSVLogger('training.log')

		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss',patience=20)
		fit = self.model.fit(self.data_set_samples, self.data_set_labels, epochs=500, verbose=2, callbacks=[early_stopping])

    #funcao para criar a rna para a abordagem hibrida
	def generateHybridModel(self):
		self.model = Sequential()
		self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))

		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		csv_logger = CSVLogger('training.log')
		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss', patience=20)

		fit = self.model.fit(self.data_set_samples, self.data_set_labels, nb_epoch=500, verbose=2, callbacks=[early_stopping])

		#obter valores da camada de saida da ultima iteracao do treinamento
		get_3rd_layer_output = K.function([self.model.layers[0].input], [self.model.layers[2].output])
		layer_output = get_3rd_layer_output([self.data_set_samples])[0]


		predictions = self.model.predict_classes(self.data_set_samples)

		return layer_output, predictions, fit

	#funcao utilizada para retornar o resultado da classificacao em termos de -1 a 1 (utilizada para a abordagem hibrida)
	def predict(self):
		predictions = self.model.predict(self.test_data_set_samples)
		return predictions

	#funcao utilizada para retornar o resultado da classificacao em 1 ou 0(utilizada para a abordagem simples)
	def predictClasses(self):
		predictions = self.model.predict_classes(self.test_data_set_samples)
		return predictions

	def setDataSet(self, data_set):
		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-2)]
		self.data_set_labels = data_set.values[:,(len(data_set.values[0])-2)]
		#print(self.data_set_samples)
		#print(self.data_set_labels)

	def setTestDataSet(self, test_data_set):
		self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		self.test_data_set_labels = test_data_set.values[:,(len(test_data_set.values[0])-2)]
		#print(self.test_data_set_samples)
		#print(self.test_data_set_labels)

	def setNumberNeuronsInputLayer(self, number):
		self.hyperparameters_setting_mode = "manual"
		self.number_neurons_input_layer = number

	def getNumberNeuronsInputLayer(self):
		return self.number_neurons_input_layer

	def setNumberNeuronsHiddenLayer(self, number):
		self.hyperparameters_setting_mode = "manual"
		self.number_neurons_hidden_layer = number

	def getNumberNeuronsHiddenLayer(self):
		return self.number_neurons_hidden_layer

	def setNumberNeuronsOutputLayer(self, number):
		self.hyperparameters_setting_mode = "manual"
		self.number_neurons_output_layer = number

	def getNumberNeuronsOutputLayer(self):
		return self.number_neurons_output_layer

	def setActivationFunctionInputLayer(self, activation_function):
		self.hyperparameters_setting_mode = "manual"
		self.activation_function_input_layer = activation_function

	def getActivationFunctionInputLayer(self):
		return self.activation_function_input_layer

	def setActivationFunctionHiddenLayer(self, activation_function):
		self.hyperparameters_setting_mode = "manual"
		self.activation_function_hidden_layer = activation_function

	def getActivationFunctionHiddenLayer(self):
		return self.activation_function_hidden_layer

	def setActivationFunctionOutputLayer(self, activation_function):
		self.hyperparameters_setting_mode = "manual"
		self.activation_function_output_layer = activation_function

	def getActivationFunctionOutputLayer(self):
		return self.activation_function_output_layer

	def setInputDimNeurons(self, number):
		self.hyperparameters_setting_mode = "manual"
		self.input_dim_neurons = number

	def getNumberNeuronsInputLayer(self):
		return self.input_dim_neurons

	def setDimInputLayer(self, dim_input_layer):
		self.hyperparameters_setting_mode = "manual"
		self.dim_input_layer = dim_input_layer

	def getDimInputLayer(self):
		return self.dim_input_layer

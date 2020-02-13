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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

class LstmModule(object):
	#conjuto de exemplos de treino
	data_set_samples = []
	#classes dos exemplos de treino
	data_set_labels = []
	#conjunto de exemplos de teste
	test_data_set_samples = []
	#classes dos exemplos de teste
	test_data_set_labels = []

	input_length = 0
	number_examples = 0

	model = None

	def __init__(self):
		print("init LSTM module")

	#funcao para criar a rna para abordagem simples
	def generateModel(self):
		self.model = Sequential()
		self.model.add(Embedding(self.number_examples, self.input_length, input_length=self.input_length))
		self.model.add(LSTM(100))
		self.model.add(Dense(1, activation='sigmoid'))
		self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(self.model.summary())

		fit = self.model.fit(self.data_set_samples, self.data_set_labels, validation_data=(self.test_data_set_samples, self.test_data_set_labels), epochs=100, verbose=2)


	#funcao utilizada para retornar o resultado da classificacao em 1 ou 0(utilizada para a abordagem simples)
	def predictClasses(self):
		predictions = self.model.evaluate(self.test_data_set_samples, self.test_data_set_labels)
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

	def setInputLength(self, input_length):
		self.input_length = input_length

	def setNumberExamples(self, number_examples):
		self.number_examples = number_examples
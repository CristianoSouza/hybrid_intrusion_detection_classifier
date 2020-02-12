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
from sklearn import svm


class SvmModule(object):
	#conjuto de exemplos de treino
	data_set_samples = []
	#classes dos exemplos de treino
	data_set_labels = []
	#conjunto de exemplos de teste
	test_data_set_samples = []
	#classes dos exemplos de teste
	test_data_set_labels = []

	clf = None

	def __init__(self):
		print("init svm module")

	#funcao para criar a rna para abordagem simples
	def generateModel(self):
		self.clf = svm.SVC()
		self.clf.fit(self.data_set_samples, self.data_set_labels)

	#funcao utilizada para retornar o resultado da classificacao em 1 ou 0
	def predictClasses(self):
		predictions = self.clf.predict(self.test_data_set_samples)
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




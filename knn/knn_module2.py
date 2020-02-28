import numpy as np
import pandas
import os
from sklearn import neighbors

class KnnModule2(object):
	#conjuto de exemplos de treino
	data_set_samples = []
	#classes dos exemplos de treino
	data_set_labels = []
	#conjunto de exemplos de teste
	test_data_set_samples = []
	#classes dos exemplos de teste
	test_data_set_labels = []
	k_neighbors = 1
	clf = None

	def __init__(self):
		print("init knn module")

	#funcao que cria a base de exemplos do KNN
	def buildExamplesBase(self):
		self.clf = neighbors.KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='kd_tree')
		#print(self.data_set_samples)
		#print(self.data_set_labels)


		#self.data_set_samples = imp.fit_transform(self.data_set_samples)
		self.clf.fit(self.data_set_samples, self.data_set_labels)

	#funcao que realiza a classificacao dos exemplos
	def run(self):
		print("-------------------------------------------------")
		
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

	def setKNeighbors(self, k_neighbors):
		self.k_neighbors = k_neighbors

	def getKNeighbors(self):
		return self.k_neighbors

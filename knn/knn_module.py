import numpy as np
import pandas
import os
from sklearn import neighbors

class KnnModule(object):
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
		self.clf = neighbors.KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='brute')
		#print(self.data_set_samples)
		#print(self.data_set_labels)

		#for x in np.nditer(self.data_set_samples):
		#	print(x)
		#print(self.data_set_samples)
		'''
		for i in range(0,10):
			for j in range(0,78):
				self.data_set_samples[i,j] = int(self.data_set_samples[i,j])
				if (self.data_set_samples[i,j] > 1000):
					self.data_set_samples[i,j] = 1000
				if (self.data_set_samples[i,j] < 0 ):
					self.data_set_samples[i,j] = 0
				print(self.data_set_samples[i,j])
		'''
		#exit()
		'''
		for i in range(0,90):
			print(str(i) +" " + str(self.data_set_labels[i]))
		'''
		'''
		if (pandas.isnull(self.data_set_samples).sum() > 0):
			print("NAN PRESENTEE")
		else:
			print("Nao exsite NAN prsente")
		'''
		#self.data_set_samples = imp.fit_transform(self.data_set_samples)
		self.clf.fit(self.data_set_samples, self.data_set_labels)

	#funcao que realiza a classificacao dos exemplos
	def run(self):
		#exit()
		print("-------------------------------------------------")
		'''for i in range(0,10):
			print(self.test_data_set_samples[i])
			for j in range(0,78):
				#self.test_data_set_samples[i,j] = int(self.test_data_set_samples[i,j])
				if (self.test_data_set_samples[i,j] > 1000):
					self.test_data_set_samples[i,j] = 1000
				if (self.test_data_set_samples[i,j] < 0 ):
					self.test_data_set_samples[i,j] = 0
				#print(self.test_data_set_samples[i,j])
		'''
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

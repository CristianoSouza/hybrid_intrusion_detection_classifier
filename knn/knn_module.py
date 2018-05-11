import numpy as np
import pandas
import os
from sklearn import neighbors

class KnnModule(object):
	data_set_samples = []
	data_set_labels = []
	test_data_set_samples = []
	test_data_set_labels = []
	k_neighbors = 1
	clf = None

	def __init__(self):
		print("init knn module")

	def buildExamplesBase(self):
		self.clf = neighbors.KNeighborsClassifier(self.k_neighbors, weights='uniform', algorithm='brute')
		self.clf.fit(self.data_set_samples, self.data_set_labels)

	def run(self):
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

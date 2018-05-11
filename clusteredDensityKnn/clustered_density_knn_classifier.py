from knn_module import KnnModule
import pandas
import os
from dataSet import DataSet

class ClusteredDensityKnnClassifier(object):

	data_set = None
	test_data_set = None
	predictions = []

	knn = None

	def __init__(self):
		print "Knn classifier"

	def run(self):
		print "RUN Knn classifier"
		self.knn.setDataSet(self.data_set)
		self.knn.setTestDataSet(self.test_data_set)
		print("aaaa")
		#print(self.test_data_set)
		self.predictions = self.knn.run()
		print("Predicao knn: ") 
		print(self.predictions)
		self.saveResults()

	def saveResults(self):
		for i in range(0,len(self.predictions)):
			self.test_data_set.set_value(i,'classe',self.predictions[i])

		DataSet.saveResults("clusteredDensityKnn", self.iteration, self.test_data_set)

	def setDataSet(self, data_set):
		self.data_set = data_set

	def getDataSet(self):
		return self.data_set

	def setTestDataSet(self, test_data_set):
		self.test_data_set = test_data_set

	def getTestDataSet(self):
		return self.test_data_set

	def setKnn(self, knn):
		self.knn = knn

	def getKnn(self):
		return self.knn

	def setIteration(self, iteration):
		self.iteration = iteration
from knn_module import KnnModule
import pandas
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dataSet import DataSet

class KnnClassifier(object):

	data_set = None
	test_data_set = None
	predictions = []
	result_path = ""
	training_time = 0
	test_time = 0
	knn = None

	def __init__(self):
		print "Knn classifier"

	def run(self):
		self.knn.setDataSet(self.data_set)
		self.knn.setTestDataSet(self.test_data_set)

		training_time_start = time.time()
		self.knn.buildExamplesBase()
		self.training_time = time.time() - training_time_start

		test_time_start = time.time()
		self.predictions = self.knn.run()
	
		self.saveResults()
		
		self.test_time = time.time() - test_time_start

	def saveResults(self):
		data_set = self.test_data_set[:] 
		for i in range(0,len(self.predictions)):
			data_set.set_value(i,'classe',self.predictions[i])
		DataSet.saveResults(self.result_path, self.iteration, data_set)

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

	def setResultPath(self, result_path):
		self.result_path = result_path

	def getTrainingTime(self):
		return self.training_time

	def getTestTime(self):
		return self.test_time
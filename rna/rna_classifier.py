from rna_module import RnaModule
import sys
import pandas
import os
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dataSet import DataSet

class RnaClassifier(object):

	data_set = None
	test_data_set = None
	rna = None
	predictions = None
	iteration = 0
	training_time = 0
	test_time = 0
	result_path = ""

	def __init__(self):
		print "aa"

	def run(self):
		training_time_start = time.time()
		print("RUN RNA classifier")
		self.rna.setDataSet(self.data_set)
		self.rna.setTestDataSet(self.test_data_set)
		
		self.rna.generateModel()
		self.training_time = time.time() - training_time_start

		test_time_start = time.time()
		self.predictions = self.rna.predictClasses()
		self.test_time = time.time() - test_time_start
		self.saveResults()

	def saveResults(self):
		for i in range(0,len(self.predictions)):
			self.test_data_set.set_value(i,'classe',self.predictions[i])

		DataSet.saveResults(self.result_path, self.iteration, self.test_data_set)	

	def setDataSet(self, data_set):
		self.data_set = data_set

	def getDataSet(self):
		return self.data_set

	def setTestDataSet(self, test_data_set):
		self.test_data_set = test_data_set

	def getTestDataSet(self):
		return self.test_data_set

	def setRna(self, rna):
		self.rna = rna

	def getRna(self):
		return self.rna

	def setIteration(self, iteration):
		self.iteration = iteration

	def setResultPath(self, result_path):
            self.result_path = result_path

	def getTrainingTime(self):
		return self.training_time

	def getTestTime(self):
		return self.test_time

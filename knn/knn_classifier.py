from knn_module import KnnModule
import pandas
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from dataSet import DataSet

class KnnClassifier(object):
	#conjunto de dados de treino
	data_set = None
	#conjunto de dados de teste
	test_data_set = None

	predictions = []

	#pasta para serem salvos os arquivos de resultados, variavel pode ser setada no arquivo main.py
	result_path = ""
	training_time = 0
	test_time = 0

	#iteracao do processo de cross-validation
	iteration = 0
	knn = None

	def __init__(self):
		print "Knn classifier"


	def run(self):
		self.knn.setDataSet(self.data_set)
		self.knn.setTestDataSet(self.test_data_set)

		training_time_start = time.time()
		#funcao para gerar a base de exemplos
		self.knn.buildExamplesBase()
		self.training_time = time.time() - training_time_start

		test_time_start = time.time()
		#funcao para realizar a classificacao dos exemplos
		self.predictions = self.knn.run()
	
		self.saveResults()
		
		self.test_time = time.time() - test_time_start
	
	#salva os resultados das classificacoes na pasta definida no arquivo main.py
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
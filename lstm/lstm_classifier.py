from rna_module import RnaModule
import sys
import pandas
import os
import time
from sklearn import svm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../hybrid_intrusion_detection_classifier")
from dataSet import DataSet

class LstmClassifier(object):
	#conjunto de dados de treino
	data_set = None
	#conjunto de dados de teste
	test_data_set = None
	
	lstm = None
	class_name = ""
	predictions = None
	
	#iteracao do processo de cross-validation
	iteration = 0


	training_time = 0
	test_time = 0

	#pasta para serem salvos os arquivos de resultados, variavel pode ser setada no arquivo main.py
	result_path = ""

	def __init__(self):
		print ("aa")

	def run(self):
		training_time_start = time.time()
		print("RUN LSTM classifier")
		self.lstm.setDataSet(self.data_set)
		self.lstm.setTestDataSet(self.test_data_set)
		
		#funcao para gerar o modelo e treina-lo
		self.lstm.generateModel()

		self.training_time = time.time() - training_time_start

		test_time_start = time.time()
		#funcao para realizar a classificacao dos exemplos
		self.predictions = self.lstm.predictClasses()
		self.test_time = time.time() - test_time_start
		self.saveResults()

	#salva os resultados das classificacoes na pasta definida no arquivo main.py
	def saveResults(self):
		for i in range(0,len(self.predictions)):
			self.test_data_set.set_value(i,self.class_name,self.predictions[i])

		DataSet.saveResults(self.result_path, self.iteration, self.test_data_set)	

	def setDataSet(self, data_set):
		self.data_set = data_set

	def getDataSet(self):
		return self.data_set

	def setTestDataSet(self, test_data_set):
		self.test_data_set = test_data_set

	def getTestDataSet(self):
		return self.test_data_set

	def setLstm(self, lstm):
		self.lstm = lstm

	def getLstm(self):
		return self.lstm

	def setIteration(self, iteration):
		self.iteration = iteration

	def setResultPath(self, result_path):
            self.result_path = result_path

	def getTrainingTime(self):
		return self.training_time

	def getTestTime(self):
		return self.test_time

	def setClass_name(self, class_name):
		self.class_name = class_name
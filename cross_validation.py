import numpy as np
import sys, os
from dataSet import DataSet
import pandas
import time
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
from hybrid_classifier import HybridClassifier
from evaluate_module import EvaluateModule
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
from rna_classifier import RnaClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")
from knn_classifier import KnnClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/clusteredKnn")
from clustered_knn_classifier import ClusteredKnnClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/clusteredDensityKnn")
from clustered_density_knn_classifier import ClusteredDensityKnnClassifier


class CrossValidation(object):
	dts = None
	#metodo utilizado para classifacao
	classifier = None

	#conjunto de dados de teste
	teste_sub_data_set = None
	#conjunto de dados de treinamento
	training_sub_data_set = None

	evaluate = None

	#numero de folds 
	k = 1

	file_path = ""

	#caminho da pasta onde serao salvos os resultados
	result_path = ""
	preprocessor = None

	def __init__(self):
		print("init")
		self.evaluate = EvaluateModule()

	def run(self):
		self.classifier.setResultPath(self.result_path)
		self.foldExecution()

	def foldExecution(self):
		i = self.iteration

		for self.iteration in range(i,(self.k+1)):
			tempo_inicio = time.time()
			self.loadTrainingData()
			self.loadTestData()

			#executa funcoes para transformacao de dados categoricos
			if self.preprocessor:
				self.preprocessor.setDataSet(self.training_sub_data_set)
				self.preprocessor.setTestDataSet(self.teste_sub_data_set)

				self.training_sub_data_set, self.teste_sub_data_set = self.preprocessor.transformCategory()

			#seta dados de treinamento e teste no classificador
			self.classifier.setDataSet(self.training_sub_data_set)
			self.classifier.setTestDataSet(self.teste_sub_data_set)

			#seta iteracao do cross no classficador
			self.classifier.setIteration(self.iteration)
			#executa o processo de treino e teste do classificador
			self.classifier.run()
			
			del(self.training_sub_data_set)
			self.loadTestData()
			#seta conjunto de dados original de teste e iteracao atual do cross-validation na classe de avaliacao
			self.evaluate.setTestDataSet(self.teste_sub_data_set)
			self.evaluate.setIteration(self.iteration)

			#verifica quel o metodo de classificacao utilziado 
			if(isinstance(self.classifier, RnaClassifier)):
				print("rna")
				self.evaluate.setResultPath( self.result_path)
			elif(isinstance(self.classifier, KnnClassifier)):
				print("knn")
				self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, ClusteredKnnClassifier)):
				print("clustered knn")
				#self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, ClusteredDensityKnnClassifier)):
				print("clustered density knn")
				#self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, HybridClassifier)):
				print("hybrid")
				self.evaluate.setResultPath( self.result_path+"final_method_classification/")

			tempo_execucao = time.time() - tempo_inicio
			self.evaluate.setTempoExecucao(tempo_execucao)
			self.evaluate.setTrainingTime(self.classifier.getTrainingTime())
			self.evaluate.setTestTime(self.classifier.getTestTime())
			#executa metodo de avaliacao
			self.evaluate.run()
	
	#carrega conjunto de treinamento de acordo coma iteracao atual do cross valiadation
	def loadTrainingData(self):
		for i in range(1,(self.k+1)):
			if( ((self.k+1) - i) != self.iteration):
				new_sub_data_set = DataSet.loadSubDataSet(self.file_path + "sub_data_set_" + str(i) + ".csv")

				if (i==1):
					self.training_sub_data_set = new_sub_data_set
				else:
					self.training_sub_data_set = DataSet.concatSubDataSet(self.training_sub_data_set, new_sub_data_set)
				del(new_sub_data_set)

		#self.training_sub_data_set = self.training_sub_data_set.reset_index()
		print(self.training_sub_data_set)

	#carrega conjunto de teste de acordo coma iteracao atual do cross valiadation
	def loadTestData(self):
		self.teste_sub_data_set = DataSet.loadSubDataSet(self.file_path + "sub_data_set_" + str((self.k+1)-self.iteration) + ".csv")
		print(self.teste_sub_data_set)
		#self.teste_sub_data_set = self.teste_sub_data_set.reset_index()
		#print(self.teste_sub_data_set)
		#exit()

	def setIteration(self, iteration):
		self.iteration = iteration

	def setClassifier(self, classifier):
		self.classifier = classifier

	def getClassifier(self):
		return classifier

	def setPreprocessor(self, preprocessor):
		self.preprocessor = preprocessor

	def getPreprocessor(self):
		return preprocessor	

	def setEvaluateModule(self, evaluate):
		self.evaluate = evaluate

	def getEvaluateModule(self):
		return evaluate	

	def setFilePath(self, file_path):
		self.file_path = file_path

	def setResultPath(self, result_path):
		self.result_path = result_path

	def setK(self, k):
		self.k = k


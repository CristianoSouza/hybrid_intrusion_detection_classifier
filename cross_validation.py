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
	classifier = None
	teste_sub_data_set = None
	training_sub_data_set = None
	evaluate = None
	method = None
	file_path = ""
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
		for self.iteration in range(i,11):
			tempo_inicio = time.time()
			self.loadTrainingData()
			self.loadTestData()

			if self.preprocessor:
				self.preprocessor.setDataSet(self.training_sub_data_set)
				self.preprocessor.setTestDataSet(self.teste_sub_data_set)

				self.training_sub_data_set, self.teste_sub_data_set = self.preprocessor.transformCategory()

			self.classifier.setDataSet(self.training_sub_data_set)
			self.classifier.setTestDataSet(self.teste_sub_data_set)

			self.classifier.setIteration(self.iteration)
			self.classifier.run()
			
			del(self.training_sub_data_set)
			self.loadTestData()
			self.evaluate.setTestDataSet(self.teste_sub_data_set)
			self.evaluate.setIteration(self.iteration)


			if(isinstance(self.classifier, RnaClassifier)):
				print("rna")
				self.evaluate.setResultPath( self.result_path)
			elif(isinstance(self.classifier, KnnClassifier)):
				print("knn")
				self.evaluate.setResultPath(self.result_path)
			elif(isinstance(self.classifier, ClusteredKnnClassifier)):
				print("clustered knn")
				#self.evaluate.setPath("clusteredKnn/")
			elif(isinstance(self.classifier, ClusteredDensityKnnClassifier)):
				print("clustered density knn")
				#self.evaluate.setPath("clusteredDensityKnn/")
			elif(isinstance(self.classifier, HybridClassifier)):
				print("hybrid")
				self.evaluate.setResultPath( self.result_path+"final_method_classification/")

			tempo_execucao = time.time() - tempo_inicio
			self.evaluate.setTempoExecucao(tempo_execucao)
			self.evaluate.setTrainingTime(self.classifier.getTrainingTime())
			self.evaluate.setTestTime(self.classifier.getTestTime())
			self.evaluate.run()
			

	def loadTrainingData(self):
		for i in range(1,11):
			if( (11 - i) != self.iteration):
				new_sub_data_set = DataSet.loadSubDataSet(self.file_path + "sub_data_set_" + str(i) + ".csv")

				if (i==1):
					self.training_sub_data_set = new_sub_data_set
				else:
					self.training_sub_data_set = DataSet.concatSubDataSet(self.training_sub_data_set, new_sub_data_set)
				del(new_sub_data_set)
		print(self.training_sub_data_set)

	def loadTestData(self):
		self.teste_sub_data_set = DataSet.loadSubDataSet(self.file_path + "sub_data_set_" + str(11-self.iteration) + ".csv")

	def setMethod(self, method):
		self.method = method

	def getMethod(self):
		return method

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


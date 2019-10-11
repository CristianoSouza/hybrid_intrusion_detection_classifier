import sys, os
import pandas 
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../knn")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../rna")

from knn_module import KnnModule
from rna_module import RnaModule

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
from dataSet import DataSet

class HybridClassifier(object):
	#iteracao do cross-validation
	iteration = 0

	data_set = None
	test_data_set = None
	
	knn = None
	rna = None
	upper_threshold = 0.7
	lower_threshold = -0.7

	intermediate_range_samples = []
	rna_classified_samples = []

	result_path = ""

	training_time = 0
	test_time = 0
	limite_faixa_sup = 0
	limite_faixa_inf = 0

	def __init__(self):
		print("init")

	def run(self):
		self.rna_classified_samples= []
		self.intermediate_range_samples = []

		self.rna.setDataSet(self.data_set)
		self.rna.setTestDataSet(self.test_data_set)
		self.knn.setDataSet(self.data_set)
		training_time_start = time.time()

		#funcao para gerar o modelo neural para a abordagem hibrida
		outputs_training, predictions, history = self.rna.generateHybridModel()
		#print (np.percentile(outputs_training,75))
		positivos = 0
		negativos = 0
		valor_negativo = 0
		valor_positivo = 0

		positivos_serie =  []
		negativos_serie =  []
		#divide os valores da camada de saida da ultima iteracao do treinamento em conjunto de positivos e de negativos
		for i in range(0,len(outputs_training)):
			if(predictions[i] == 0 ):
				negativos = negativos + 1
				valor_negativo = valor_negativo + outputs_training[i]
				negativos_serie.append(outputs_training[i])
			elif(predictions[i] == 1):
				positivos = positivos + 1
				valor_positivo = valor_positivo + outputs_training[i]
				positivos_serie.append(outputs_training[i])

		#cria base de exemplos do KNN
		self.knn.buildExamplesBase()
		self.training_time = time.time() - training_time_start

		list_position_rna_classified_samples = []
		list_position_intermediate_range_samples = []

		test_time_start = time.time()
		#inicia teste
		#realiza classificacao atraves da RNA
		self.predictions_rna = self.rna.predict()
		self.test_time = time.time() - test_time_start
                
		tamanho_predicao = len(self.predictions_rna)
		tamanho_data_set = len(self.test_data_set.values)
		#posicao do atributo "classe" no vetor
		posicao_classe = len(self.test_data_set.values[0]) - 2
  		
		if (self.verifyClassesPredictions(predictions) == True):
			#define os limites superiores e inferiores de acordo com os valores de percentil para definir a faixa intermediaria (valores de percentil sao setados no arquivo main.py)
			self.upper_threshold = np.percentile(positivos_serie,self.percentil_faixa_sup)
			self.lower_threshold = np.percentile(negativos_serie,(100 - self.percentil_faixa_inf))

			#verifica se valor esta dentro dos limites ou fora
			for i in range(0,len(self.predictions_rna)):
				print(self.predictions_rna[i])
				if(self.predictions_rna[i] > (self.upper_threshold) ):
					#print("CLASSIFICACAO CONFIAVEL!")
					#realiza as modificacoes no dataframe dos exemplos originais de teste de acordo com a classificacao da RNA
					self.test_data_set.set_value(i, 'classe', 1)
				elif( self.predictions_rna[i] < (self.lower_threshold)):
					#print("CLASSIFICACAO CONFIAVEL!")
					#realiza as modificacoes no dataframe dos exemplos originais de teste de acordo com a classificacao da RNA
					self.test_data_set.set_value(i, 'classe', 0)
				else:
					#print("FAIXA INTERMEDIARIA!")
					#adiciona exemplos em um vetor de exemplos classificados como intermediarios
					self.intermediate_range_samples.append(self.test_data_set.values[i,:])
					list_position_intermediate_range_samples.append(i)
			del(self.predictions_rna)

			#cria um dataframe de exemplos classificados pela RNA
			dataframe_rna_classified_samples = pandas.DataFrame(
					data= self.rna_classified_samples,
					index= list_position_rna_classified_samples,
					columns= self.test_data_set.columns)

			print(dataframe_rna_classified_samples)

			#salva os resultados gerados pela RNA
			DataSet.saveResults( self.result_path + "rna_classification/", self.iteration, dataframe_rna_classified_samples)
			del(dataframe_rna_classified_samples)
			del(list_position_rna_classified_samples)
		else:
			for i in range(0,len(self.predictions_rna)):
				self.intermediate_range_samples.append(self.test_data_set.values[i,:])
				list_position_intermediate_range_samples.append(i)

		#cria um dataframe de exemplos classificados como intermediarios	
		dataframe_intermediate_range_samples = pandas.DataFrame(
			data= self.intermediate_range_samples,
			index= list_position_intermediate_range_samples,
			columns= self.test_data_set.columns)

		#seta o dataframe de exemplos intermediarios como conjunto de teste para o KNN
		self.knn.setTestDataSet(dataframe_intermediate_range_samples)

		#salva os exemplos enviados para o KNN apenas para possivel identificacao posterior
		DataSet.saveResults( self.result_path + "knn_classification/", self.iteration, dataframe_intermediate_range_samples)
		
		test_time_start = time.time()
		#executa o KNN para classificar os exemplos do conjunto de teste
		self.predictions_knn = self.knn.run()
		self.test_time = self.test_time + (time.time() - test_time_start)
		
		del(self.data_set)
		del(dataframe_intermediate_range_samples)


		#realiza as modificacoes no dataframe dos exemplos originais de teste de acordo com a classificacao do KNN
		for i in range(0,len(self.predictions_knn)):
			self.test_data_set.set_value(list_position_intermediate_range_samples[i], 'classe', self.predictions_knn[i])

		#salva o data frame modificado como o resultado final
		DataSet.saveResults( self.result_path + "final_method_classification/", self.iteration, self.test_data_set)
		del(self.test_data_set)

	def verifyClassesPredictions(self, predictions):
		sair = 0
		for i in range(0,len(predictions)):
			if (predictions[i] == 0):
				sair = 1
			elif ((predictions[i] == 1) & (sair == 1 )):
				return True
		return False

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

	def setRna(self, rna):
		self.rna = rna

	def getRna(self):
		return self.rna

	def setIteration(self, iteration):
		self.iteration = iteration

	def setUpperThreshold(self, upper_threshold):
		self.upper_threshold = upper_threshold

	def getUpperThreshold(self):
		return self.upper_threshold

	def setLowerThreshold(self, lower_threshold):
		self.lower_threshold = lower_threshold

	def getLowerThreshold(self):
		return lower_threshold

	def setResultPath(self, result_path):
		self.result_path = result_path

	def getTrainingTime(self):
		return self.training_time

	def getTestTime(self):
		return self.test_time

	def setLimiteFaixaSup(self, limite_faixa):
		self.limite_faixa_sup = limite_faixa

	def setLimiteFaixaInf(self, limite_faixa):
		self.limite_faixa_inf = limite_faixa
	
	def setPercentilFaixaSup(self, limite_faixa):
		self.percentil_faixa_sup = limite_faixa

	def setPercentilFaixaInf(self, limite_faixa):
		self.percentil_faixa_inf = limite_faixa

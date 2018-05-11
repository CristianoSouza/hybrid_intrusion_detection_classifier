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
		outputs_training, predictions, history = self.rna.generateHybridModel()
		#print (np.percentile(outputs_training,75))
		positivos = 0
		negativos = 0
		valor_negativo = 0
		valor_positivo = 0

		positivos_serie =  []
		negativos_serie =  []
		for i in range(0,len(outputs_training)):
			if(predictions[i] == 0 ):
				negativos = negativos + 1
				valor_negativo = valor_negativo + outputs_training[i]
				negativos_serie.append(outputs_training[i])
			elif(predictions[i] == 1):
				positivos = positivos + 1
				valor_positivo = valor_positivo + outputs_training[i]
				positivos_serie.append(outputs_training[i])

		arquivo = open('RNA_loss' + str(self.iteration) + '.txt', 'w') 
		texto = """RNA LOSS""" + str(history.history['loss']) + """ -- """
		texto = """RNA ACC""" + str(history.history['acc']) + """ -- """
		arquivo.write(texto) 
		arquivo.close()


		self.knn.buildExamplesBase()
		self.training_time = time.time() - training_time_start

	
		list_position_rna_classified_samples = []
		list_position_intermediate_range_samples = []

		test_time_start = time.time()
		self.predictions_rna = self.rna.predict()
		self.test_time = time.time() - test_time_start
                
		#print(len(self.predictions_rna))
		tamanho_predicao = len(self.predictions_rna)
		tamanho_data_set = len(self.test_data_set.values)
		posicao_classe = len(self.test_data_set.values[0]) - 2
  		
  		arquivo_up = open('INFO_up_' + str(self.iteration) + '.txt', 'w')
  		arquivo_inf = open('INFO_inf_' + str(self.iteration) + '.txt', 'w') 
  		arquivo_inter = open('INFO_inter_' + str(self.iteration) + '.txt', 'w') 

  		texto_up = """INICIO
  		"""
  		texto_inf = """INICIO
  		"""
  		texto_inter = """INICIO
  		"""
		if (self.verifyClassesPredictions(predictions) == True):
			print("!")

			mediana_positivos = np.percentile(positivos_serie,50)
			mediana_negativos = np.percentile(negativos_serie,50)
			print (mediana_positivos)
			print (mediana_negativos)

			quartile_sup = np.percentile(positivos_serie,self.percentil_faixa_sup)
			print (quartile_sup)
			quartile_inf = np.percentile(negativos_serie,(100 - self.percentil_faixa_inf))
			print (quartile_inf)
			#exit()

			self.upper_threshold = quartile_sup
			self.lower_threshold = quartile_inf


			for i in range(0,len(self.predictions_rna)):
				print(self.predictions_rna[i])
				if(self.predictions_rna[i] > (self.upper_threshold) ):
					#print("CLASSIFICACAO CONFIAVEL!")
					texto_up += """[""" + str(i) + """] -- [""" + str(self.test_data_set.values[i,(posicao_classe+1)]) + """]= """ + str(self.predictions_rna[i]) + """ -- """ + str(self.test_data_set.values[i,posicao_classe]) + """
				"""
					self.test_data_set.set_value(i, 'classe', 1)
				elif( self.predictions_rna[i] < (self.lower_threshold)):
					#print("CLASSIFICACAO CONFIAVEL!")
					texto_inf += """[""" + str(i) + """] -- [""" + str(self.test_data_set.values[i,(posicao_classe+1)]) + """]= """ + str(self.predictions_rna[i]) + """ -- """ + str(self.test_data_set.values[i,posicao_classe]) + """
				"""
					self.test_data_set.set_value(i, 'classe', 0)
				else:
					#print("FAIXA INTERMEDIARIA!")
					texto_inter += """[""" + str(i) + """] -- [""" + str(self.test_data_set.values[i,(posicao_classe+1)]) + """]= """ + str(self.predictions_rna[i]) + """ -- """ + str(self.test_data_set.values[i,posicao_classe]) + """
				"""
					self.intermediate_range_samples.append(self.test_data_set.values[i,:])
					list_position_intermediate_range_samples.append(i)


			arquivo_up.write(texto_up) 
			arquivo_up.close()
			arquivo_inf.write(texto_inf) 
			arquivo_inf.close()
			arquivo_inter.write(texto_inter) 
			arquivo_inter.close()

			del(self.predictions_rna)

			dataframe_rna_classified_samples = pandas.DataFrame(
					data= self.rna_classified_samples,
					index= list_position_rna_classified_samples,
					columns= self.test_data_set.columns)

			print(dataframe_rna_classified_samples)

			DataSet.saveResults( self.result_path + "rna_classification/", self.iteration, dataframe_rna_classified_samples)
			del(dataframe_rna_classified_samples)
			del(list_position_rna_classified_samples)
		else:
			for i in range(0,len(self.predictions_rna)):
				self.intermediate_range_samples.append(self.test_data_set.values[i,:])
				list_position_intermediate_range_samples.append(i)


		dataframe_intermediate_range_samples = pandas.DataFrame(
			data= self.intermediate_range_samples,
			index= list_position_intermediate_range_samples,
			columns= self.test_data_set.columns)

		self.knn.setTestDataSet(dataframe_intermediate_range_samples)
		DataSet.saveResults( self.result_path + "knn_classification/", self.iteration, dataframe_intermediate_range_samples)

		test_time_start = time.time()
		self.predictions_knn = self.knn.run()
		self.test_time = self.test_time + (time.time() - test_time_start)
		
		del(self.data_set)
		del(dataframe_intermediate_range_samples)

		for i in range(0,len(self.predictions_knn)):
			self.test_data_set.set_value(list_position_intermediate_range_samples[i], 'classe', self.predictions_knn[i])

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

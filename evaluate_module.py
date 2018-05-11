import sys, os
from dataSet import DataSet
from preprocessor import Preprocessor
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "hybrid")

from hybrid_classifier import HybridClassifier

class EvaluateModule(object):

	number_false_positives = 0
	number_false_negatives = 0
	number_true_positives = 0
	number_true_negatives = 0
	classes = None
	test_data_set = None
	iteration = 0
	result_path = ""
	total_samples= 0
	acc_samples= 0
	err_samples=0
	tempo_execucao = 0
	classifier = None
	training_time = 0 
	test_time = 0

	def __init__(self):
		print("init")

	def run(self):
		self.number_false_positives = 0
		self.number_false_negatives = 0
		self.number_true_positives = 0
		self.number_true_negatives = 0		
		self.total_samples = 0
		self.acc_samples = 0
		self.err_samples = 0

		print("RUN evaluate method")
		result_dataframe = DataSet.loadResult(self.result_path , self.iteration)
	
		print(result_dataframe)

		self.classes = Preprocessor.getClassesPerColumns(self.test_data_set,'classe')

		acc_classes = []
		err_classes = []
		print(self.test_data_set.values)
		print(result_dataframe.values)
		print(len(self.test_data_set.values))
		#exit()
		if(len(self.classes) <=2 ):
			posicao_classe = len(result_dataframe.values[0]) -2
			for i in range(0,len(result_dataframe.values)):
				self.total_samples+= 1
				print("Real: " + str(self.test_data_set.values[i,posicao_classe]) + " -- predito: " + str(result_dataframe.values[i,posicao_classe]))
				if(self.test_data_set.values[i,posicao_classe] == '0' or self.test_data_set.values[i,posicao_classe] == 0 ):
					if (result_dataframe.values[i,posicao_classe] == 0 or result_dataframe.values[i,posicao_classe] == '0'):
						#print("FALSO E CLASSIFICOU COMO FALSO")
						self.number_true_negatives+=1
						self.acc_samples+=1
					else:
						#print("FALSO E CLASSIFICOU COMO VERDADEIRO")
						self.number_false_positives+=1
						self.err_samples+=1 

				elif(self.test_data_set.values[i,posicao_classe] == '1' or self.test_data_set.values[i,posicao_classe] == 1):
					if (result_dataframe.values[i,posicao_classe] == 1 or result_dataframe.values[i,posicao_classe] == '1'):
						#print("VERDADEIRO E CLASSIFICOU COMO VERDADEIRO")
						self.number_true_positives+=1
						self.acc_samples+=1
					else:
						#print("VERDADEIRO E CLASSIFICOU COMO FALSO")
						self.number_false_negatives+=1
						self.err_samples+=1 

			arquivoMatriz = open(self.result_path + 'Matriz.txt', 'a+') 
			textoMatriz = str(self.number_true_positives) + """,""" + str(self.number_false_positives) + ""","""+ str(self.number_false_negatives) + """,""" + str(self.number_true_negatives) + """
"""  
			arquivoMatriz.write(textoMatriz) 
			arquivoMatriz.close()

			arquivoTempo = open(self.result_path + 'tempo.txt', 'a+') 
			textoTempo = str(self.tempo_execucao) + """,""" + str(self.training_time) + ""","""+ str(self.test_time) +  """
"""  
			arquivoTempo.write(textoTempo) 
			arquivoTempo.close()

			arquivoInfos = open(self.result_path + 'infos.txt', 'a+') 
			textoInfos = str(self.total_samples) + """,""" + str(self.acc_samples) + ""","""+ str(self.err_samples) +  """,""" + str((100/float(self.total_samples)) * self.acc_samples) + """,""" +  str((100/float(self.total_samples)) * self.err_samples) + """
"""  
			arquivoInfos.write(textoInfos) 
			arquivoInfos.close()


			arquivo = open(self.result_path + 'final_info_' + str(self.iteration) + '.txt', 'w') 
			texto = """		MATRIZ DE CONFUSAO
	             Predicao      
			 ATAQUE    NORMAL  
		   |--------||--------|
	ATAQUE |   """ + str(self.number_true_positives) + """    ||   """+ str(self.number_false_negatives) + """    |
		   |--------||--------|
	NORMAL |   """+ str(self.number_false_positives) + """    ||   """+ str(self.number_true_negatives) + """    |
		   |--------||--------|
			"""
		else:
			posicao_classe = len(result_dataframe.values[0]) -2
			arquivo = open(self.result_path + 'final_info_' + str(self.iteration) + '.txt', 'w') 
			texto = """		MATRIZ DE CONFUSAO
	             Predicao      
			 ACC    ERR  
|--------||--------|

"""	
			for i in range(0,len(self.classes)):
				acc_classes.append(0)
				err_classes.append(0)
			for i in range(0,len(result_dataframe.values)):
				print(self.test_data_set.values[i,posicao_classe])
				print(result_dataframe.values[i,posicao_classe])
				#exit()
				self.total_samples+= 1
				print("Real: " + str(self.test_data_set.values[i,posicao_classe]) + " -- predito: " + str(result_dataframe.values[i,posicao_classe]))

				if(self.test_data_set.values[i,posicao_classe] == "normal"):
					if (result_dataframe.values[i,posicao_classe] ==  "normal"):
						acc_classes[0]+=1
						self.acc_samples+=1
					else:
						err_classes[0]+=1
						self.err_samples+=1 
				elif(self.test_data_set.values[i,posicao_classe] != "normal"):
					if(result_dataframe.values[i,posicao_classe] != "normal"):
						acc_classes[1]+=1
						self.acc_samples+=1
					else:
						err_classes[1]+=1
						self.err_samples+=1 

			for i in range(0,len(acc_classes)):		
				texto+=  """ 	|   """ + str(acc_classes[i]) + """    || 	""" + str(err_classes[i]) + """ | 		
|--------||--------|
"""
		texto+= """TOTAL DE EXEMPLOS: """ + str(self.total_samples) + """ 	|   
|--------||--------|
"""
		texto+= """TOTAL DE EXEMPLOS CORRETOS: """ + str(self.acc_samples) + """ 	|   
|--------||--------|
"""
		texto+= """TOTAL DE EXEMPLOS ERRADOS: """ + str(self.err_samples) + """ 	|   
|--------||--------|
"""
		texto+= """PORCENTAGEM ACERTOS: """ + str((100/float(self.total_samples)) * self.acc_samples) + """ 	|   
|--------||--------|
"""
		texto+= """PORCENTAGEM ERROS: """ + str((100/float(self.total_samples)) * self.err_samples) + """ 	|   
|--------||--------|
"""			
		texto+="""TEMPO DE EXECUCAO: """ + str(self.tempo_execucao) + """  ||| 
"""
		texto+="""TEMPO DE TREINO: """ + str(self.training_time) + """  ||| 
"""
		texto+="""TEMPO DE TESTE: """ + str(self.test_time) + """  ||| 
"""
		if (DataSet.checkPathBoolean(self.result_path + "../knn_classification/")):
			data_set_knn = DataSet.loadSubDataSet( self.result_path + "../knn_classification/cross_"+ str(self.iteration) + "_final_result.csv") 
			texto+= """Exemplos submetidos a segunda classificacao: """ + str(len(data_set_knn))
			arquivoKNN = open(self.result_path + 'KNN.txt', 'a+') 
			textoKNN= str(len(data_set_knn)) + """
"""  
			arquivoKNN.write(textoKNN) 
			arquivoKNN.close()
			print(len(data_set_knn))
		arquivo.write(texto) 
		arquivo.close()
	
	def setTestDataSet(self, test_data_set):
		self.test_data_set = test_data_set

	def setResultPath(self, result_path):
		self.result_path = result_path

	def setNumberFalsePositives(self, number_false_positives):
		self.number_false_positives = number_false_positives

	def getNumberFalsePositives(self):
		return number_false_positives

	def setNumberFalseNegatives(self, number_false_negatives):
		self.number_false_negatives = number_false_negatives

	def getNumberFalseNegatives(self):
		return number_false_negatives

	def setNumberTruePositives(self, number_true_positives):
		self.number_true_positives = number_true_positives

	def getNumberTruePositives(self):
		return number_true_positives

	def setNumberTrueNegatives(self, number_true_negatives):
		self.number_true_negatives = number_true_negatives

	def getNumberTrueNegatives(self):
		return self.number_true_negatives

	def setIteration(self, iteration):
		self.iteration = iteration

	def setClasses(self, classes):
		self.classes = classes

	def getClasses(self):
		return self.classes
        
	def setClassifier(self, classifier):
		self.classifier = classifier

	def setTempoExecucao(self, tempo_execucao):
		self.tempo_execucao = tempo_execucao

	def setTrainingTime(self, training_time):
		self.training_time = training_time

	def setTestTime(self, test_time):
		self.test_time = test_time
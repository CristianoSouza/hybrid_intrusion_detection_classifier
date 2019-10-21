import pandas
import os
import random

class DataSet(object):

	file_name = ""
	test_file_name = ""
	dataframe_data_set = []
	partition_size = 0
	number_folds = 10
	file_path = ""

	def __init__(self):
		print ("init")

	#funcao para carregar base de dados
	def loadData(self, number):
		self.number_folds = number
		self.dataframe_data_set = pandas.read_csv(self.file_name)
		#funcao para particionar a base de dados em K conjuntos de modo aleatorio
		self.partitionDataSet()

	def selectExamples(self):
		lista = list(range(0, self.dataframe_data_set.shape[0]))

		tamanho = len(lista)
		#print (list
		#print (tamanho)
		arquivo = open( "saidaaa.csv", 'a')
		arquivo.write("\nEntrou.... vai colocar posicao ORIGINAL")
		arquivo.close()
		for a in range(0,tamanho):
			#self.dataframe_data_set.scdet_value(a,'po', 15)
			self.dataframe_data_set.loc[a, 'posicaoOriginal'] = int(a)
        
        arquivo = open( "saidaaa.csv", 'a')
        arquivo.write("\ncolocou posicao ORIGINAL")
        arquivo.close()

		#print(self.dataframe_data_set)
		directory = os.path.dirname(self.file_path)
		if not os.path.exists(directory):
			print("nom ecsiste")
			os.makedirs(directory)
		else:
			print("ecsiste")

		arquivo = open( "saidaaa.csv", 'a')
		arquivo.write("\nmontou a listaaaaa \n VAI COMECAR A PARTICIONAR A BASE...")
		arquivo.close()
		data_set = []

		data_set_posicoes = []
		len_attributes = len(self.dataframe_data_set.values[0,:])
		for i in range(0,10):
			sub_data_set = []
			#print(self.partition_size)
			posicoes = random.sample(lista,int(self.partition_size))
			#print("--")
			#print(posicoes)
			len_posicoes = len(posicoes)

			arquivo = open( self.file_path + "sub_data_set_" + str(i+1) + ".csv", 'w') 
			for k in range(0,len(self.dataframe_data_set.columns)):
				texto = str(self.dataframe_data_set.columns[k])
				arquivo.write(texto) 
				if(k+1 < len(self.dataframe_data_set.columns)):
					arquivo.write(""",""") 
				else:
					arquivo.write("""
""")	

			for j in range(0,len_posicoes):
				#print(":::::")
				#print(lista)
				#print(posicoes[j])
				lista.remove(int(posicoes[j]))
				texto = ""
				#print(str(i) + " - " + str(j))
				linha = self.dataframe_data_set.values[posicoes[j],:]
				for k in range(0,len_attributes):
					linha[k] = round(float(linha[k]), 2)
					texto += str(linha[k])
					if(k+1 < len_attributes): 
						texto += ""","""
					else:
						texto +="""
"""  
				arquivo.write(texto) 
			arquivo.close()


		if lista:
			for i in range(0,len(lista)):
				arquivo = open( self.file_path + "sub_data_set_" + str(i+1) + ".csv", 'a') 
				texto = ""
				linha = self.dataframe_data_set.values[lista[i],:]
				for k in range(0,len_attributes):
					texto += str(linha[k])
					if(k+1 < len_attributes):
						texto += ""","""
					else:
						texto +="""
"""  
				arquivo.write(texto) 
				arquivo.close()

	#funcao para particionar a base de dados em K conjuntos de modo aleatorio
	def partitionDataSet(self):
		self.partition_size = (self.dataframe_data_set.shape[0] / self.number_folds)
		self.selectExamples()

	@classmethod
	def loadSubDataSet(self, file_name):
		sub_dataframe_data_set = pandas.read_csv(file_name)
		return sub_dataframe_data_set

	@classmethod
	def concatSubDataSet(self, data_frame1, data_frame2):
		frames = [data_frame1, data_frame2]
		return pandas.concat(frames)

	@classmethod
	def loadResult(self, result_path, iteration):
		return pandas.read_csv(str(result_path) + "cross_" + str(iteration) + "_final_result.csv")

	@classmethod
	def saveResults(self, result_path, iteration,  data_frame):
		directory = os.path.dirname(result_path)
		if not os.path.exists(directory):
			print("nao existe")
			os.makedirs(directory)
		else:
			print("exists")	

		data_frame.to_csv( str(result_path) + "cross_" + str(iteration) + "_final_result.csv", sep=',', index=False)

	@classmethod
	def checkPath(self, file_path):
		directory = os.path.dirname(self.file_path)
		if not os.path.exists(directory):
			print("nao existe")
			os.makedirs(directory)
		else:
			print("exists")	
	
	@classmethod
	def checkPathBoolean(self, file_path):
		print(file_path)
		directory = os.path.dirname(file_path)
		print(directory)
		
		if (os.path.exists(file_path)):
			return True
		else:
			return False

	def setFileName(self, file_name):
		self.file_name = file_name

	def setFilePath(self, file_path):
		self.file_path = file_path

	def getDataSet(self):
		return self.data_set

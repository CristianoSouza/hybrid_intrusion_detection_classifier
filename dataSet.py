import pandas
import os
import random

class DataSet(object):

	file_name = ""
	test_file_name = ""
	dataframe_data_set = []
	partition_size = 0
	file_path = ""

	def __init__(self):
		print "init"

	def setFileName(self, file_name):
		self.file_name = file_name

	def setFilePath(self, file_path):
		self.file_path = file_path

	def loadData(self):
		self.dataframe_data_set = pandas.read_csv("bases/" + self.file_name)
		self.partitionDataSet()

	def getDataSet(self):
		return self.data_set

	def selectExamples(self):
		lista = range(0, self.dataframe_data_set.shape[0])

		tamanho = len(lista)
		print lista
		print tamanho
		for a in range(0,tamanho):
			#self.dataframe_data_set.scdet_value(a,'po', 15)
			self.dataframe_data_set.loc[a, 'posicaoOriginal'] = a

		directory = os.path.dirname(self.file_path)
		if not os.path.exists(directory):
			print("nom ecsiste")
			os.makedirs(directory)
		else:
			print("ecsiste")	

		data_set = []
		data_set_posicoes = []
		len_attributes = len(self.dataframe_data_set.values[0,:])
		for i in range(0,10):
			sub_data_set = []
			posicoes = random.sample(lista,self.partition_size)
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
				lista.remove(posicoes[j])
				texto = ""
				print(str(i) + " - " + str(j))
				linha = self.dataframe_data_set.values[posicoes[j],:]
				for k in range(0,len_attributes):
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

	def partitionDataSet(self):
		self.partition_size = (self.dataframe_data_set.shape[0] / 10)
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



 

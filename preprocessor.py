from sklearn.preprocessing import LabelEncoder
import keras.preprocessing.text

class Preprocessor(object):

	columns_category = []
	data_set = None
	test_data_set = None
	classes = {}

	def __init__(self):
		print ("init")

	#Funcao para transfmormar atributos categoricos em inteiros
	def transformCategory(self):
		le = LabelEncoder()
		for col in self.columns_category:
			le.fit(self.data_set[col])
			self.classes[col]= le.classes_
			
			self.data_set[col] = le.fit_transform(self.data_set[col])
			self.test_data_set[col] = le.fit_transform(self.test_data_set[col])

		print("CLASSESS:")
		print(self.classes)
		return self.data_set, self.test_data_set 

	#funcao para obter quantidade de classes existentes em determinado atributo
	@classmethod
	def getClassesPerColumns(self, data_set, column):
		le = LabelEncoder()
		le.fit(data_set[column])
		return le.classes_

	def setColumnsCategory(self, columns_category):
		self.columns_category = columns_category

	def getColumnsCategory(self):
		return self.columns_category

	def setDataSet(self, data_set):
		self.data_set = data_set

	def getDataSet(self):
		return self.data_set

	def setTestDataSet(self, test_data_set):
		self.test_data_set = test_data_set

	def getTestDataSet(self):
		return self.test_data_set
import sys, os
import pandas
from preprocessor import Preprocessor
from dataSet import DataSet

dataframe_data_set = pandas.read_csv("../../Bases/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
#dataframe_data_set = pandas.read_csv("../../Bases/MachineLearningCVE/teste_ddos.csv")

tamanho = dataframe_data_set.shape[0]
print(tamanho)
print(dataframe_data_set.shape[1])
print(dataframe_data_set.values[1,78])


for a in range(0,tamanho):
	if (dataframe_data_set.values[a,78] == 'BENIGN' ):
		#print("BENIGN")
		dataframe_data_set.loc[a, ' Label'] = '0' 
	else:
		dataframe_data_set.loc[a, ' Label2'] = '1' 

'''
for ix in dataframe_data_set.index:
    dataframe_data_set.loc[ix, " Label"] = 'My New Value'
'''
#dataframe_data_set.to_csv("../../Bases/MachineLearningCVE/teste_ddos_BINARY.csv", sep=',', index=False)
dataframe_data_set.to_csv( "../../Bases/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos_BINARY.pcap_ISCX.csv", sep=',', index=False)

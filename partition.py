import os
from dataSet import DataSet


arquivo = open( "saidaaa.csv", 'w') 
arquivo.write("CHAMOU PARTITION") 
arquivo.close()

dts = DataSet()

print("Iniciando particionamento....")
arquivo = open( "saidaaa.csv", 'a')
arquivo.write("Iniciando particionamento....A") 
arquivo.close()
#pasta pra salvar

#dts.setFilePath("../../Bases/MachineLearningCVE/teste/")
dts.setFilePath("../../Bases/MachineLearningCVE/DoS_56att/")

#caminho e nome do arquivo
#dts.setFileName("../../Bases/MachineLearningCVE/teste_ddos_BINARY.csv")
dts.setFileName("../../Bases/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos_BINARY_56att.pcap_ISCX.csv")
print("chamando load")
arquivo = open( "saidaaa.csv", 'a')
arquivo.write("chamando load...A")
arquivo.close()
dts.loadData(10)

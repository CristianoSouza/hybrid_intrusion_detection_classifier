from dataSet import DataSet

dts = DataSet()

print("Iniciando particionamento....")
#pasta pra salvar

#dts.setFilePath("../../Bases/MachineLearningCVE/teste/")
dts.setFilePath("../../Bases/MachineLearningCVE/DoS_56att/")

#caminho e nome do arquivo
#dts.setFileName("../../Bases/MachineLearningCVE/teste_ddos_BINARY.csv")
dts.setFileName("../../Bases/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos_BINARY_56att.pcap_ISCX.csv")
print("chamando load")
dts.loadData(10)

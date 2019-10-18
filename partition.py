
import sys, os
import pandas as pd
from preprocessor import Preprocessor
from dataSet import DataSet
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")

from cross_validation import CrossValidation
from preprocessor import Preprocessor
from dataSet import DataSet
from knn_classifier import KnnClassifier
from rna_classifier import RnaClassifier
from hybrid_classifier import HybridClassifier
from rna_module import RnaModule
from knn_module import KnnModule
from evaluate_module import EvaluateModule


dts = DataSet()

print("Iniciando particionamento....")
#pasta pra salvar
#dts.setFilePath("../../Bases/MachineLearningCVE/teste/")
dts.setFilePath("../../Bases/MachineLearningCVE/DoS_56att/")

#caminho e nome do arquivo
#dts.setFileName("../../Bases/MachineLearningCVE/teste_ddos_BINARY.csv")
dts.setFileName("../../Bases/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos_BINARY_56att.pcap_ISCX.csv")

dts.loadData(10)

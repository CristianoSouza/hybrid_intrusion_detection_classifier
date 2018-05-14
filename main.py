
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
dts.setFilePath("bases/sub_bases_nslkdd_20attribute/")
#dts.setFileName("base_iris.csv")
#dts.setFileName("SmallTrainingSet.csv")
##dts.setFileName("winequality-red.csv")
#dts.setFileName("NSL_KDD-master/20PercentTrainingSet.csv")
dts.setFileName("NSL_KDD-master/KDDTrain+binary_class.csv")
#dts.setFileName("NSL_KDD-master/SmallTrainingSet.csv")
#dts.setFileName("NSL_KDD-master/SmallTrainingSetFiveClass.csv")

#dts.setFileName("../../KDDCUP99/kddcup10%.csv")


#print("load data")
#ts.loadData(10)


#CONFIGURACAO DO KNN
knn = KnnModule()
knn.setKNeighbors(1)
knn_classifier = KnnClassifier()
knn_classifier.setKnn(knn)

#CONFIGURACAO DA REDE NEURAL 
rna = RnaModule()
rna.setNumberNeuronsImputLayer(41)
rna.setActivationFunctionImputLayer("tanh")
rna.setImputDimNeurons(41)
rna.setNumberNeuronsHiddenLayer(41)
rna.setActivationFunctionHiddenLayer("tanh")
rna.setNumberNeuronsOutputLayer(1)
rna.setActivationFunctionOutputLayer("tanh")
rna_classifier = RnaClassifier()
rna_classifier.setRna(rna)

#METODO HIBRIDO 
hybrid_classifier = HybridClassifier()
hybrid_classifier.setPercentilFaixaSup(25)
hybrid_classifier.setPercentilFaixaInf(100)
hybrid_classifier.setRna(rna)
hybrid_classifier.setKnn(knn)


#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['protocol_type','service','flag'])

evaluate = EvaluateModule()

cross = CrossValidation()
#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)
cross.setK(10)
cross.setPreprocessor(preprocessor)
#cross.setFilePath("bases/sub_bases_20_nslkdd/")
#cross.setFilePath("bases/sub_bases_train+_nslkdd/")
#cross.setFilePath("bases/sub_bases_nslkdd_tcp_attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_12attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_20attribute/")
#cross.setFilePath("bases/sub_bases_SmallTrainingSet/")
cross.setFilePath("bases/sub_bases_small_training_set1000/")

#cross.setResultPath("results/faixa_hibrido/")
cross.setResultPath("results/teste_casa/")

#cross.setClassifier(rna_classifier)
#cross.setClassifier(knn_classifier)
#cross.setClassifier(clustered_knn_classifier)
#cross.setClassifier(clustered_density_knn_classifier)
cross.setClassifier(hybrid_classifier)

cross.setEvaluateModule(evaluate)
cross.run()

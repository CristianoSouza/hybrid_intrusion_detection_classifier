import sys, os
import pandas as pd
from preprocessor import Preprocessor
from dataSet import DataSet
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/svm")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rf")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/naive_bayes")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/lstm")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/dt")

from cross_validation import CrossValidation
from preprocessor import Preprocessor
from dataSet import DataSet
from knn_classifier import KnnClassifier
from svm_classifier import SvmClassifier
from naive_bayes_classifier import NaiveBayesClassifier
from rf_classifier import RfClassifier
from rna_classifier import RnaClassifier
from lstm_classifier import LstmClassifier
from hybrid_classifier import HybridClassifier
from decision_tree_classifier import DecisionTreeClassifier
from rna_module import RnaModule
from knn_module import KnnModule
from svm_module import SvmModule
from rf_module import RfModule
from decision_tree_module import DecisionTreeModule
from lstm_module import LstmModule
from naive_bayes_module import NaiveBayesModule
from evaluate_module import EvaluateModule


#CONFIGURACAO DO KNN
knn = KnnModule()
knn.setKNeighbors(1)
knn_classifier = KnnClassifier()
knn_classifier.setKnn(knn)

#CONFIGURACAO DO SVM
svm = SvmModule()
svm_classifier = SvmClassifier()
svm_classifier.setSvm(svm)

#CONFIGURACAO DO RF
rf = RfModule()
rf_classifier = RfClassifier()
rf_classifier.setRf(rf)

#CONFIGURACAO DO RF
dt = DecisionTreeModule()
dt_classifier = DecisionTreeClassifier()
dt_classifier.setDecisionTree(dt)

#CONFIGURACAO DA NAIVEBAYES
naive_bayes = NaiveBayesModule()
naive_bayes_classifier = NaiveBayesClassifier()
naive_bayes_classifier.setNaiveBayes(naive_bayes)

#CONFIGURACAO DO LSTM
lstm = LstmModule()
lstm.setInputLength(20)
lstm.setNumberExamples(1000)
lstm_classifier = LstmClassifier()
lstm_classifier.setLstm(lstm)


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
#preprocessor.setColumnsCategory(['service','flag'])

evaluate = EvaluateModule()

cross = CrossValidation()
#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)
cross.setK(10)
cross.setPreprocessor(preprocessor)
#cross.setFilePath("bases/sub_bases_20_nslkdd/")
#icross.setFilePath("bases/sub_bases_train+_nslkdd/")
#cross.setFilePath("bases/sub_bases_nslkdd_tcp_attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_12attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_20attribute/")
#cross.setFilePath("bases/sub_bases_SmallTrainingSet/")
#cross.setFilePath("bases/sub_bases_small_training_set1000/")

#cross.setResultPath("results/faixa_hibrido/")
#cross.setFilePath("../../Bases/MachineLearningCVE/DoS/")
#cross.setFilePath("../../Bases/NSL-KDD/bases/attribute_selection/sub_bases_iris/")
#cross.setFilePath("../../Bases/NSL-KDD/bases/attribute_selection/sub_bases_nslkdd_20attribute/")
cross.setFilePath("../../Bases/NSL-KDD/bases/attribute_selection/sub_bases_nslkdd_complete/")
#cross.setResultPath("../results_ann-knn_cicids2017_ddos/completa/svm/")
#cross.setResultPath("../results_ann-knn_NSL-KDD/20att/dt/")
#cross.setResultPath("../results_iris/completa/naive_bayes/")
#cross.setResultPath("../results_iris/completa/dt/")
cross.setResultPath("../NOVOSEXPERIMENTOS/NSL-KDD/completa/hybP25N1-1/")
#cross.setClassifier(rna_classifier)
#cross.setClassifier(knn_classifier)
#cross.setClassifier(svm_classifier)
#cross.setClassifier(rf_classifier)
#cross.setClassifier(naive_bayes_classifier)
#cross.setClassifier(lstm_classifier)
#cross.setClassifier(dt_classifier)
#cross.setClassifier(clustered_knn_classifier)
#cross.setClassifier(clustered_density_knn_classifier)
cross.setClassifier(hybrid_classifier)

cross.setClass_name('classe')
#cross.setClass_name(' Label')

cross.setEvaluateModule(evaluate)
cross.run()

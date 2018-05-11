
import sys, os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../rna")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../knn")

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
dts.setFilePath("bases/sub_bases/")

#CONFIGURACAO DA REDE NEURAL 
rna = RnaModule()
rna.setNumberNeuronsImputLayer(30)
rna.setActivationFunctionImputLayer("tanh")
rna.setImputDimNeurons(30)
rna.setNumberNeuronsHiddenLayer(31)
rna.setActivationFunctionHiddenLayer("tanh")
rna.setNumberNeuronsOutputLayer(1)
rna.setActivationFunctionOutputLayer("tanh")
rna_classifier = RnaClassifier()
rna_classifier.setRna(rna)

#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['protocol_type','service','flag'])

evaluate = EvaluateModule()

cross = CrossValidation()

#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)

cross.setPreprocessor(preprocessor)

cross.setFilePath("../../bases/sub_bases_nslkdd_30_attribute/")

cross.setResultPath("../../results/30_attribute/rna_oculta_31_time/")

cross.setClassifier(rna_classifier)

cross.setEvaluateModule(evaluate)

cross.run()


import sys, os
sys.path.append( os.path.dirname(os.path.realpath(__file__))+ "/../../../rna")
print (os.path.dirname(os.path.realpath(__file__)))
from rna_classifier import RnaClassifier
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../../knn")
from knn_classifier import KnnClassifier
from cross_validation import CrossValidation
from preprocessor import Preprocessor
from hybrid_classifier import HybridClassifier
from rna_module import RnaModule
from knn_module import KnnModule
from evaluate_module import EvaluateModule
from dataSet import DataSet

dts = DataSet()
dts.setFilePath("bases/sub_bases/")

#CONFIGURACAO DA REDE NEURAL 
rna = RnaModule()
rna.setNumberNeuronsImputLayer(41)
rna.setActivationFunctionImputLayer("tanh")
rna.setImputDimNeurons(41)
rna.setNumberNeuronsHiddenLayer(42)
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

cross.setFilePath("../../bases/sub_bases_train+_nslkdd/")

cross.setResultPath("../../results/complete_attribute/rna_oculta_42_time/")

cross.setClassifier(rna_classifier)

cross.setEvaluateModule(evaluate)

cross.run()

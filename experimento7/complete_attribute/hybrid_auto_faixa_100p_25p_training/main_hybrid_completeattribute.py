
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
rna.setNumberNeuronsHiddenLayer(42)
rna.setActivationFunctionHiddenLayer("tanh")
rna.setNumberNeuronsOutputLayer(1)
rna.setActivationFunctionOutputLayer("tanh")
rna_classifier = RnaClassifier()
rna_classifier.setRna(rna)

#METODO HIBRIDO 
hybrid_classifier = HybridClassifier()
#hybrid_classifier.setLowerThreshold(-0.95)
#hybrid_classifier.setUpperThreshold(0.95)
hybrid_classifier.setPercentilFaixaSup(25)
hybrid_classifier.setPercentilFaixaInf(100)
#hybrid_classifier.setLimiteFaixaInf(-0.2)
#hybrid_classifier.setLimiteFaixaSup(0.1)
hybrid_classifier.setRna(rna)
hybrid_classifier.setKnn(knn)


#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['protocol_type','service','flag'])

evaluate = EvaluateModule()

cross = CrossValidation()

#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)

cross.setPreprocessor(preprocessor)

cross.setFilePath("../../bases/sub_bases_train+_nslkdd/")

cross.setResultPath("../../results/complete_attribute/hybrid_auto_faixa_100p_25p_training_info/")
cross.setClassifier(hybrid_classifier)

cross.setEvaluateModule(evaluate)

cross.run()

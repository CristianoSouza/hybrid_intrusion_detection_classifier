
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

#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['service','flag'])

evaluate = EvaluateModule()

cross = CrossValidation()

#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)

cross.setPreprocessor(preprocessor)

cross.setFilePath("../../bases/sub_bases_nslkdd_12_attribute/")

cross.setResultPath("../../results/12_attribute/knn_brute_time/")

cross.setClassifier(knn_classifier)

cross.setEvaluateModule(evaluate)

cross.run()

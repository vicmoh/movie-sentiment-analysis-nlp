# Author: Vicky Mohammad
# Description: File containing the main to
# to run the program.

# Import my files
import sys
from sk_learn import SkLearn
from sk_learn import Classifier
# Import lib
from sklearn import svm
from sklearn import datasets
# Import data analysis
from data_analysis import DataAnalysis

_PATH_DATA = './assets/review_polarity/data'
_PATH_TEST = './assets/review_polarity/test'


class Main:
    def __init__(self):
        """Class to run a specific programs."""
        super().__init__()
        # Main.runSelectedModels()
        Main.dataAnalysis()

    @staticmethod
    def dataAnalysis():
        DataAnalysis.readAllFilesInFolder(_PATH_DATA + '/pos/*.txt')
        DataAnalysis.readAllFilesInFolder(_PATH_DATA + '/neg/*.txt')
        DataAnalysis.readAllFilesInFolder(_PATH_TEST + '/pos/*.txt')
        DataAnalysis.readAllFilesInFolder(_PATH_TEST + '/neg/*.txt')
        DataAnalysis.printResult()

    @staticmethod
    def runSelectedModels():
        sk = SkLearn()
        print('\nRunning logistic regression...')
        sk.run(classifier=Classifier.logisticRegression)
        print('\nRunning random forest classifier...')
        sk.run(classifier=Classifier.randomForestClassifier)
        print('\nRunning linear SVC...')
        sk.run(classifier=Classifier.linearSVC)

    @staticmethod
    def run():
        print('Running...')
        SkLearn(classifier=Classifier.linearSVC)


Main()

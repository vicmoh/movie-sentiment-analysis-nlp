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


class Main:
    def __init__(self):
        """Class to run a specific programs."""
        super().__init__()
        Main.runSelectedModels()

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

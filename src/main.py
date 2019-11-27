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
        print('\nRunning logistic regression...')
        SkLearn(classifier=Classifier.logisticRegression)
        print('\nRunning random forest classifier...')
        SkLearn(classifier=Classifier.randomForestClassifier)
        print('\nRunning linear SVC...')
        SkLearn(classifier=Classifier.linearSVC)


Main()

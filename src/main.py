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

# For the debug statement
_FILE_EXAMPLE = './assets/review_polarity/txt_sentoken/pos/cv000_29590.txt'
_PATH = './assets/review_polarity/txt_sentoken/'


class Main:
    def __init__(self):
        """Class to run a specific programs."""
        super().__init__()
        Main.runAll()

    @staticmethod
    def runAll():
        # print('\nRunning logistic regression...')
        # SkLearn(classifier=Classifier.logisticRegression)
        # print('\nRunning random forest classifier...')
        # SkLearn(classifier=Classifier.randomForestClassifier)
        # print('\nRunning k nearest neighbors...')
        # SkLearn(classifier=Classifier.kNearestNeighbors)
        # print('\nRunning linear SVC...')
        # SkLearn(classifier=Classifier.linearSVC)
        print('\nRunning multi class SVC...')
        SkLearn(classifier=Classifier.nuSVC)
        # print('\nRunning gaussian naive bayes...')
        # SkLearn(classifier=Classifier.gaussianNB)
        # print('\nRunning multinomial naive bayes...')
        # SkLearn(classifier=Classifier.multiNB)
        # print('\nRunning complement naive bayes...')
        # SkLearn(classifier=Classifier.complementNB)
        # print('\nRunning bernoulli naive bayes...')
        # SkLearn(classifier=Classifier.BernoulliNB)

    @staticmethod
    def run():
        SkLearn()


Main()

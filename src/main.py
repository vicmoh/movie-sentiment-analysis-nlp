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
        Main.runDataAnalysis()
        # Main.runSelectedModels()

    @staticmethod
    def runDataAnalysis():
        posData = DataAnalysis(_PATH_DATA + '/pos/*.txt')
        negData = DataAnalysis(_PATH_DATA + '/neg/*.txt')
        print('\n---------- Positive data analysis ----------')
        posData.printResult()
        print('\n---------- Negative data analysis ----------')
        negData.printResult()
        print('\n---------- Positive and negative data analysis ----------')
        posData.printCombinedResultWith(negData)

    @staticmethod
    def runSelectedModels():
        sk = SkLearn()
        # Run with TFxIDF vectorizer
        print('\n---------- Running with TFxIDF feature selection ----------')
        for kFoldSize in [5, 10]:
            for f_size in [500, 1000, 1500, 2000, 2500, 3000]:
                print('\nRunning logistic regression...')
                sk.run(classifier=Classifier.logisticRegression,
                       featureSize=f_size, isTfidfVec=True, num_kFold=kFoldSize)
                print('\nRunning random forest classifier...')
                sk.run(classifier=Classifier.randomForestClassifier,
                       featureSize=f_size, isTfidfVec=True, num_kFold=kFoldSize)
                print('\nRunning linear SVC...')
                sk.run(classifier=Classifier.linearSVC,
                       featureSize=f_size, isTfidfVec=True, num_kFold=kFoldSize)
        # Run with count vectorizer
        print('\n---------- Running with count vectorizer feature selection ----------')
        for kFoldSize in [5, 10]:
            for f_size in [500, 1000, 1500, 2000, 2500, 3000]:
                print('\nRunning logistic regression...')
                sk.run(classifier=Classifier.logisticRegression,
                       featureSize=f_size, isCountVec=True, num_kFold=kFoldSize)
                print('\nRunning random forest classifier...')
                sk.run(classifier=Classifier.randomForestClassifier,
                       featureSize=f_size, isCountVec=True, num_kFold=kFoldSize)
                print('\nRunning linear SVC...')
                sk.run(classifier=Classifier.linearSVC,
                       featureSize=f_size, isCountVec=True, num_kFold=kFoldSize)

    @staticmethod
    def run():
        print('Running...')
        SkLearn(classifier=Classifier.linearSVC)


Main()

import numpy as np
import pandas as pd
import os
import re
import sklearn.datasets as _SkData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

_PATH = './assets/review_polarity/txt_sentoken'
_TRAIN_PATH = './src/example/movie_data/full_train.txt'
_TEST_PATH = './src/example/movie_data/full_test.txt'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"


class SkLearn():
    pathFolder = None
    reviews_train = None
    reviews_test = None
    posData = None
    negData = None

    def __init__(self, pathFolder=None):
        """Initialize the natural language process. 
        @pathFolder is the path to the data containing the 'pos' and 
        'neg' folder."""
        super().__init__()
        self.pathFolder = _PATH
        if pathFolder is not None:
            self.pathFolder = pathFolder
        self.posData = _SkData.load_files(self.pathFolder, categories='pos')
        self.negData = _SkData.load_files(self.pathFolder, categories='neg')
        self.preProcess(self.posData['data'])

    @staticmethod
    def splitDataForTrain(posData, negData):
        """Split data for training and development sets.
        """
        trainPos, valPos, trainNeg, valNeg = train_test_split(
            posData, negData, train_size=0.85, random_state=0)
        trainData = trainPos.concat(trainNeg)
        testData = valPos.concat(valNeg)

        train = ''
        test = ''
        for each in trainData:
            train += re.sub('[ \r\n]+|[\n]+', '', each)
        for each in testData:
            test += re.sub('[ \r\n]+|[\n]+', '', each)

        return train, test

    @staticmethod
    def countVectorizer(trainData, testData):
        """Count vectorizer."""
        cv = CountVectorizer(binary=True).fit()
        cv.fit(trainData)
        X = cv.transform(trainData)
        X_test = cv.transform(testData)

    def preProcess(self, data):
        _FUNC_DEBUG = True
        preData = re.sub(_REPLACE_NO_SPACE, '', data)
        if _FUNC_DEBUG:
            print(preData)
        return preData

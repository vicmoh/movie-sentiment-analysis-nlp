from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For printing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.datasets as _SkData
import numpy as _Numpy
import pandas as _Panda
import os
import re
import nltk
nltk.download('stopwords')

_PATH = './assets/review_polarity/txt_sentoken'
_TRAIN_PATH = './src/example/movie_data/full_train.txt'
_TEST_PATH = './src/example/movie_data/full_test.txt'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"


class SkLearn():
    pathFolder = ''
    stopwords = stopwords.words('english')

    def __init__(self, pathFolder=None):
        super().__init__()
        X, y = SkLearn.loadData(pathFolder)
        docs = SkLearn.preprocess(X)
        X = SkLearn.bagOfWords(docs, self.stopwords)
        X = SkLearn.tfidfProcess(X, docs, self.stopwords)
        X_train, X_test, y_train, y_test = SkLearn.trainSplit(X, y)
        y_pred = Classifier.randomForestClassifier(X_train, X_test, y_train)
        SkLearn.printResult(y_test, y_pred)

    @staticmethod
    def loadData(pathFolder=None):
        if pathFolder is None:
            pathFolder = _PATH
        reviewFile = _SkData.load_files(pathFolder)
        return reviewFile.data, reviewFile.target

    @staticmethod
    def preprocess(data):
        docs = []
        stemmer = WordNetLemmatizer()
        for sen in range(0, len(data)):
            # Remove all the special characters
            doc = re.sub(r'\W', ' ', str(data[sen]))
            # remove all single characters
            doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
            # Remove single characters from the start
            doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc)
            # Substituting multiple spaces with single space
            doc = re.sub(r'\s+', ' ', doc, flags=re.I)
            # Removing prefixed 'b'
            doc = re.sub(r'^b\s+', '', doc)
            # Converting to Lowercase
            doc = doc.lower()
            # Lemmatization
            doc = doc.split()
            doc = [stemmer.lemmatize(word) for word in doc]
            doc = ' '.join(doc)
            docs.append(doc)
        return docs

    @staticmethod
    def bagOfWords(docs, stopwords):
        return CountVectorizer(
            max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords
        ).fit_transform(docs).toarray()

    @staticmethod
    def tfidfProcess(X, docs, stopwords):
        # Transform
        X = TfidfTransformer().fit_transform(X).toarray()
        # Convert
        X = TfidfVectorizer(
            max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords
        ).fit_transform(docs).toarray()
        return X

    @staticmethod
    def trainSplit(data, target):
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, train_size=0.85, random_state=0)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def printResult(y_test, y_pred):
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))


class Classifier():
    @staticmethod
    def randomForestClassifier(X_train, X_test, y_train):
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        return y_pred

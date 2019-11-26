# Author: Vicky Mohammad
# Description: File for processing the sentiment analysis,
# the file contains SkLearn class for the processing,
# and the Classifier for the model being used.

# Stop words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import snowball
from nltk.stem import PorterStemmer
from nltk.stem import lancaster
# Sklearn libs
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # For printing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
# Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import metrics
import sklearn.datasets as _SkData
# Others
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

    def __init__(self, pathFolder=None, classifier=None):
        """Scikit learn class for processing the sentiment analysis to
        determine if it is negative or positive data. @pathFolder is the
        folder of the data containing 'pos' and 'neg' folder of the txt files.
        @classifier is callback function that takes in classifier function in
        the Classifier() object, by default if none it will run logistic regression."""
        super().__init__()
        # Check if the classifier is empty
        if classifier is None:
            classifier = Classifier.logisticRegression
        # Load the data and preprcess the word
        X, y = SkLearn.loadData(pathFolder)
        docs = SkLearn.preprocess(X)
        # N-gram bag of words and tf idf the data
        X, cv = SkLearn.bagOfWords(docs, self.stopwords)
        X = SkLearn.tfidfProcess(X, docs, self.stopwords)
        # Split the data for training and testing
        X_train, X_test, y_train, y_test = SkLearn.trainSplit(X, y)
        # Run classifier and print the result
        y_pred = classifier(X_train, X_test, y_train)
        SkLearn.printResult(y_test, y_pred)

    @staticmethod
    def loadData(pathFolder=None):
        """Load data from the @pathFolder that contain 
        'pos' and 'neg' folder of the txt files.
        Returns data, target."""
        if pathFolder is None:
            pathFolder = _PATH
        reviewFile = _SkData.load_files(pathFolder)
        return reviewFile.data, reviewFile.target

    @staticmethod
    def preprocess(data):
        """Preprocess the @data and return documents that
        has been processed through stemmed."""
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
        """Create a bag of words, using n-gram model, to determine
        the document and term frequency. Return the data and count vectorizer object
        that is used to process the bag of words."""
        cv = CountVectorizer(max_features=1500, min_df=5,
                             max_df=0.7, ngram_range=(1, 3), stop_words=stopwords)
        return cv.fit_transform(docs).toarray(), cv

    @staticmethod
    def tfidfProcess(data, docs, stopwords):
        """Go through the TFxIDF process similar for the
        document and term frequency. Return the data proccessed."""
        # Transform
        data = TfidfTransformer().fit_transform(data).toarray()
        # Convert
        data = TfidfVectorizer(
            max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords
        ).fit_transform(docs).toarray()
        return data

    @staticmethod
    def trainSplit(data, target):
        """Split the data for training and testing.
        Return X data for train, X data for text, y data for train,
        y data for test."""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, train_size=0.85, random_state=0)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def printResult(y_test, y_pred):
        """Print the y test and y predicted data of the
        confusion matrix, classification report, and accuracy score."""
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))


class Classifier():
    """Function containing static methods used as a lambda calls
    for the classification for the SkLearn."""

    @staticmethod
    def randomForestClassifier(X_train, X_test, y_train):
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        return y_pred

    @staticmethod
    def logisticRegression(X_train, X_test, y_train, C=1):
        model = LogisticRegression(C=C, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                   penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                   verbose=0, warm_start=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    @staticmethod
    def passiveAggressive(X_train, X_test, y_train):
        return PassiveAggressiveClassifier(
            C=1.0, average=False, fit_intercept=True, loss='hinge',
            n_jobs=1, shuffle=True, warm_start=False
        ).fit(X_train, y_train).predict(X_test)

    @staticmethod
    def ridgeClassifier(X_train, X_test, y_train):
        return RidgeClassifier(
            alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
            max_iter=None, normalize=False, random_state=None, solver='auto'
        ).fit(X_train, y_train).predict(X_test)

    @staticmethod
    def standardSVC(X_train, X_test, y_train):
        return SVC().fit(X_train, y_train).predict(X_test)

    @staticmethod
    def linearSVC(X_train, X_test, y_train, C=1):
        svm = LinearSVC(class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                        verbose=0, C=C)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        return y_pred

    @staticmethod
    def nuSVC(X_train, X_test, y_train):
        return NuSVC().fit(X_train, y_train).predict(X_test)

    @staticmethod
    def kNearestNeighbors(X_train, X_test, y_train):
        knn = KNeighborsClassifier(n_neighbors=500)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        return y_pred

    @staticmethod
    def gaussianNB(X_train, X_test, y_train):
        return GaussianNB().fit(X_train, y_train).predict(X_test)

    @staticmethod
    def multiNB(X_train, X_test, y_train):
        return MultinomialNB().fit(X_train, y_train).predict(X_test)

    @staticmethod
    def complementNB(X_train, X_test, y_train):
        return ComplementNB().fit(X_train, y_train).predict(X_test)

    @staticmethod
    def BernoulliNB(X_train, X_test, y_train):
        return BernoulliNB().fit(X_train, y_train).predict(X_test)

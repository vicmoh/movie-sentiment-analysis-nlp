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
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import metrics
from sklearn.metrics import f1_score
import sklearn.datasets as _SkData
# Others
import numpy as _Numpy
import pandas as _Panda
import os
import re
import nltk
nltk.download('stopwords')

_PATH_DATA = './assets/review_polarity/data'
_PATH_TEST = './assets/review_polarity/test'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"


class SkLearn():
    dataFolderPath = ''
    stopwords = stopwords.words('english')

    def __init__(self, dataFolderPath=None, testFolderPath=None, classifier=None):
        """Scikit learn class for processing the sentiment analysis to
        determine if it is negative or positive data. @dataFolderPath is the
        folder of the data containing 'pos' and 'neg' folder of the txt files.
        @classifier is callback function that takes in classifier function in
        the Classifier() object, by default if none it will run logistic regression."""
        super().__init__()
        # Check if the classifier is empty
        if classifier is None:
            classifier = Classifier.logisticRegression
        if dataFolderPath is None:
            dataFolderPath = _PATH_DATA
        if testFolderPath is None:
            testFolderPath = _PATH_TEST
        # Load the data and preprcess the word
        X, y = SkLearn.loadData(dataFolderPath)
        X_test, y_test = SkLearn.loadData(testFolderPath)
        docs = SkLearn.preprocess(X)
        # N-gram bag of words and tf idf the data
        X, model = SkLearn.bagOfWords(docs, self.stopwords)
        X, model = SkLearn.tfidfProcess(X, docs, self.stopwords)
        # Split the data for training and testing
        print(SkLearn.kFold(X, y))
        # Validate with the test file

    @staticmethod
    def loadData(pathFolder):
        """Load data from the @pathFolder that contain 
        'pos' and 'neg' folder of the txt files.
        Returns data, target."""
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
        vec = CountVectorizer(max_features=1500, min_df=5,
                              max_df=0.7, ngram_range=(1, 3), stop_words=stopwords)
        return vec.fit_transform(docs).toarray(), vec

    @staticmethod
    def tfidfProcess(data, docs, stopwords):
        """Go through the TFxIDF process similar for the
        document and term frequency. Return the data proccessed."""
        # Transform
        data = TfidfTransformer().fit_transform(data).toarray()
        # Convert
        vec = TfidfVectorizer(max_features=1500, min_df=5,
                              max_df=0.7, stop_words=stopwords)
        data = vec.fit_transform(docs).toarray()
        return data, vec

    @staticmethod
    def trainSplit(data, target):
        """Split the data for training and testing.
        Return X data for train, X data for text, y data for train,
        y data for test."""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, train_size=0.85, random_state=0)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def kFold(data, target, num_splits=5, classifier=None):
        """Function for k-fold. Return the list of scores."""
        if classifier is None:
            classifier = Classifier.logisticRegression
        kf = StratifiedKFold(n_splits=num_splits)
        scores = list()
        for train_index, test_index in kf.split(data, target):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]
            y_pred, model = classifier(X_train, X_test, y_train)
            scores.append(model.score(X_test, y_test))
            SkLearn.printResult(y_test, y_pred)
        return scores

    @staticmethod
    def CustomKFoldExample(X_digit, y_digit, num_folds=5, sk_classifier=None):
        if sk_classifier is None:
            sk_classifier = LinearSVC
        import numpy as np
        X_folds = np.array_split(X_digit, num_folds)
        y_folds = np.array_split(y_digit, num_folds)
        scores = list()
        for k in range(num_folds):
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            y_train = list(y_folds)
            y_test = y_train.pop(k)
            y_train = np.concatenate(y_train)
            scores.append(sk_classifier().fit(
                X_train, y_train).score(X_test, y_test))
        print(scores)

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
        model = RandomForestClassifier(n_estimators=1000, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    @staticmethod
    def logisticRegression(X_train, X_test, y_train, C=1):
        model = LogisticRegression(C=C, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                                   penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                                   verbose=0, warm_start=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    @staticmethod
    def linearSVC(X_train, X_test, y_train, C=1):
        model = LinearSVC(class_weight=None, dual=True, fit_intercept=True,
                          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                          verbose=0, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

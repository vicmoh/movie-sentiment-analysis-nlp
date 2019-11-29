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
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
    X = None
    y = None
    X_test = None
    y_test = None
    dataFolderPath = ''
    stopwords = stopwords.words('english')
    docs = None

    def __init__(self, dataFolderPath=None, testFolderPath=None):
        """Scikit learn class for processing the sentiment analysis to
        determine if it is negative or positive data. @dataFolderPath is the
        folder of the data containing 'pos' and 'neg' folder of the txt files.
        @classifier is callback function that takes in classifier function in
        the Classifier() object, by default if none it will run logistic regression."""
        super().__init__()
        self.__initData(dataFolderPath=dataFolderPath,
                        testFolderPath=testFolderPath)

    def __initData(self, dataFolderPath=None, testFolderPath=None):
        """Load the data and save for processing."""
        if dataFolderPath is None:
            dataFolderPath = _PATH_DATA
        if testFolderPath is None:
            testFolderPath = _PATH_TEST
        # Load the data
        self.X, self.y = SkLearn.__loadData(dataFolderPath)
        self.X_test, self.y_test = SkLearn.__loadData(testFolderPath)
        print('\nPreprocessing...')
        self.docs = SkLearn.preprocess(self.X)

    def run(self, classifier=None, num_kFold=5, featureSize=500, isTfidfVec=False, isCountVec=False):
        """Run the model specified."""
        # Check if the classifier is empty
        if classifier is None:
            classifier = Classifier.logisticRegression
        # Feature selection
        X, y, X_test, y_test = self.X, self.y, self.X_test, self. y_test
        X, cv = SkLearn.__featureSelectionProcess(
            X, self.docs, self.stopwords, featureSize=featureSize)
        # Cross validation
        print('Feature size: ', featureSize)
        print('Cross validating with ' +
              str(num_kFold) + ' Stratified K-Fold...')
        scores, fMeasures = SkLearn.__kFold(
            X, y, num_splits=num_kFold, classifier=classifier)
        print('Accuracy scores: ', scores)
        print('F-measure scores: ', fMeasures)
        print('Average score:', _Numpy.average(scores))
        print('Average f-measure: ', _Numpy.average(fMeasures))
        # final test with the 15 percent
        print('Running with 15% testing sets...')
        X_train, X_test, y_train, y_test = self.__trainSplit(X, y)
        y_pred, old_model = classifier(X_train, X_test, y_train)
        accuracy_score = old_model.score(X_test, y_test)
        f1_measure = f1_score(y_test, y_pred)
        print('Model score: ', accuracy_score)
        print('Model f-measure score: ', f1_measure)
        SkLearn.printTerms(cv, old_model)
        predictions = cross_val_predict(old_model, X, y, cv=5)
         
        

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
    def printTerms(cv, final_model):
        """Print the term and accuracy of the models."""
        print('\nPrinting terms...')
        try:
            feature_to_coef = {
                word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])
            }
            for highest_positives in sorted(
                    feature_to_coef.items(),
                    key=lambda x: x[1],
                    reverse=True)[:20]:
                print(highest_positives)
            for highest_negatives in sorted(
                    feature_to_coef.items(),
                    key=lambda x: x[1])[:20]:
                print(highest_negatives)
        except:
            print('Could not print term results.')

    @staticmethod
    def printResult(y_test, y_pred):
        """Print the y test and y predicted data of the
        confusion matrix, classification report, and accuracy score."""
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))

    @staticmethod
    def __loadData(pathFolder):
        """Load data from the @pathFolder that contain
        'pos' and 'neg' folder of the txt files.
        Returns data, target."""
        reviewFile = _SkData.load_files(pathFolder, shuffle=True)
        return reviewFile.data, reviewFile.target

    @staticmethod
    def __bagOfWords(docs, stopwords, featureSize=500):
        """Create a bag of words, using n-gram model, to determine
        the document and term frequency. Return the data and count vectorizer object
        that is used to process the bag of words."""
        vec = CountVectorizer(max_features=featureSize, min_df=5,
                              max_df=0.7, ngram_range=(1, 3), stop_words=stopwords)
        return vec.fit_transform(docs).toarray(), vec

    @staticmethod
    def __tfidfProcess(data, docs, stopwords, featureSize=500):
        """Go through the TFxIDF process similar for the
        document and term frequency. Return the data proccessed."""
        # Transform
        data = TfidfTransformer().fit_transform(data).toarray()
        # Convert
        vec = TfidfVectorizer(max_features=featureSize, min_df=5,
                              max_df=0.7, stop_words=stopwords)
        data = vec.fit_transform(docs).toarray()
        return data, vec

    @staticmethod
    def __featureSelectionProcess(X, docs, stopwords, isCountVec=False, isTfidfVec=False, featureSize=None):
        # Check type of selection being used
        if isCountVec is False and isTfidfVec is False:
            isCountVec = True
            isTfidfVec = True
        # N-gram bag of words and tf idf the data
        X, model = None, None
        if isCountVec:
            X, model = SkLearn.__bagOfWords(
                docs, stopwords, featureSize=featureSize)
        if isTfidfVec:
            X, model = SkLearn.__tfidfProcess(
                X, docs, stopwords, featureSize=featureSize)
        return X, model

    @staticmethod
    def __trainSplit(data, target):
        """Split the data for training and testing.
        Return X data for train, X data for text, y data for train,
        y data for test."""
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, train_size=0.85, random_state=0)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def __kFold(data, target, num_splits=5, classifier=None):
        """Function for k-fold. Return the list of scores."""
        if classifier is None:
            classifier = Classifier.logisticRegression
        kf = StratifiedKFold(n_splits=num_splits)
        scores = list()
        fMeasures = list()
        for train_index, test_index in kf.split(data, target):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]
            y_pred, model = classifier(X_train, X_test, y_train)
            scores.append(model.score(X_test, y_test))
            fMeasures.append(f1_score(y_test, y_pred))
        return scores, fMeasures

    @staticmethod
    def __customKFoldExample(X_digit, y_digit, num_folds=5, sk_classifier=None):
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


class Classifier():
    """Function containing static methods used as a lambda calls
    for the classification for the SkLearn."""

    @staticmethod
    def multiLayerPerceptron(X_train, X_test, y_train):
        model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(5, 2),
                              learning_rate='constant', learning_rate_init=0.001,
                              max_iter=200, momentum=0.9, n_iter_no_change=10,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    @staticmethod
    def multinomialNB(X_train, X_test, y_train):
        model = MultinomialNB(alpha=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

    @staticmethod
    def KNeighbors(X_train, X_test, y_train):
        model = KNeighborsClassifier(n_neighbors=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model

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

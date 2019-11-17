# Author: Vicky Mohammad
# Description: This file is used to process the movie
# classification, using other classes.

from util.util import Util
import re as _regex

# Default folder for the positive path
_DEFAULT_STOP_WORDS_PATH = './assets/stopwords.txt'
_POS_FOLDER = './assets/review_polarity/txt_sentoken/pos/*.txt'
_PUNC_REGEX = '[^\w0-9]'  # "[.;:!\'?,\"()\[\]-\+=#\$\*]"
_WHITE_SPACE = "[\r\n]+|[\n]+|[\t]+|[ ]+"


class Process:
    filePath = ''
    stopWords = []
    docs = []

    def __init__(self, filePath=None):
        super().__init__()
        print('Processing...')
        if filePath == None:
            self.filePath = filePath
        else:
            self.filePath = _POS_FOLDER

    @staticmethod
    def isWordInList(word, listOfWords):
        """Check if @word is in @listOfWords. Return true if it is."""
        if word is None or listOfWords is None:
            return False
        for each in listOfWords:
            if each.strip() == word.strip():
                return True
        return False

    @staticmethod
    def isPunc(word):
        """Remove punctuation.
        @word string to be checked.
        Return true if it is punctuation else return false."""
        if len(word) is not 1:
            return False
        if _regex.match(_PUNC_REGEX, word) is not None:
            return True
        return False

    @staticmethod
    def removePuncInLine(sentence, andListOfWords=None):
        """Remove punctuation in sentence.
        @sentence to be edited.
        @andListOfWords to be removed.
        Return the edited string."""
        FUNC_DEBUG = False
        newSen = ''
        splitted = _regex.compile(_WHITE_SPACE).split(sentence)
        for word in splitted:
            if (FUNC_DEBUG):
                print('word: ' + word)
            if word.strip()[0] is not '\'':
                word = _regex.sub(_PUNC_REGEX, '', word.strip())
            # Remove punctuation
            if (Process.isPunc(word) or Process.isWordInList(word, andListOfWords)) is not True:
                newSen += word + ' '
        return _regex.sub('[ \t]+', ' ', newSen.strip())

    def readStopWords(self, filePath=None):
        """Read the stop words from the assets folder.
        Return list of stop words."""
        path = _DEFAULT_STOP_WORDS_PATH
        if filePath is not None:
            path = filePath
        self.stopWords = Util.readFile(path)
        return self.stopWords

    def run(self):
        """Function to run and process the movie."""
        print('Running...')
        self.readStopWords()
        self.docs = Util.readAllFilesInFolder(
            _POS_FOLDER,
            numFiles=3,
            eachLineCallback=lambda line: Process.removePuncInLine(line, andListOfWords=self.stopWords))

    def toString(self):
        """To string function to for printing the docs."""
        toBeReturn = ''
        for doc in self.docs:
            for line in doc:
                toBeReturn += line + '\n'
        return toBeReturn

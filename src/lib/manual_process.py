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


class ManualProcess:
    filePath = ''
    stopWords = {}
    docs = []

    def __init__(self, filePath=None):
        super().__init__()
        print('Processing...')
        if filePath == None:
            self.filePath = filePath
        else:
            self.filePath = _POS_FOLDER

    @staticmethod
    def isWordInList(word, mapOfWords):
        """Check if @word is in @mapOfWords. Return true if it is."""
        if word is None or mapOfWords is None:
            return False
        if word in mapOfWords:
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
    def removePuncInLine(sentence, wordsToBeRemoved=None):
        """Remove punctuation in sentence.
        @sentence to be edited.
        @wordsToBeRemoved is a map.
        Return the edited string."""
        FUNC_DEBUG = False
        newSen = ''
        splitted = _regex.compile(_WHITE_SPACE).split(sentence)
        for word in splitted:
            if (FUNC_DEBUG):
                print('word: ' + word)
            # Remove punctuation
            if (Process.isPunc(word) or Process.isWordInList(word, wordsToBeRemoved)) is not True:
                newSen += word + ' '
        return _regex.sub('[ \t]+', ' ', newSen.strip())

    def readStopWords(self, filePath=None):
        """Read the stop words from the assets folder
        or from @filePath.
        Return map of stop words."""
        path = _DEFAULT_STOP_WORDS_PATH
        if filePath is not None:
            path = filePath
        stopWordsList = Util.readFile(path)
        for each in stopWordsList:
            self.stopWords[each] = True
        return self.stopWords

    def run(self):
        """Function to run and process the movie."""
        print('Running...')
        self.readStopWords()
        self.docs = Util.readAllFilesInFolder(
            _POS_FOLDER,
            numFiles=2,
            eachLineCallback=lambda line: Process.removePuncInLine(line, wordsToBeRemoved=self.stopWords))

    def toString(self):
        """To string of what's in the docs."""
        toBeReturn = ''
        for doc in self.docs:
            for line in doc:
                toBeReturn += line + '\n'
        return toBeReturn

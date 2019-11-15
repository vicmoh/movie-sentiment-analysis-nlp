# This file is used to process the movie
# classification, using other classes.

from util.util import Util
import re as _regex

# Default folder for the positive path
_POS_FOLDER = './assets/review_polarity/txt_sentoken/pos/*.txt'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"
_WHITE_SPACE = "[\r\n]+|[\n]+|[\t]+|[ ]+"


class Process:
    filePath = ''
    docs = []

    def __init__(self, filePath=None):
        super().__init__()
        print('Processing...')
        if filePath == None:
            self.filePath = filePath
        else:
            self.filePath = _POS_FOLDER

    @staticmethod
    def isPunc(word):
        """Remove punctuation
        @word string to be checked
        Return true if it is punctuation else return false"""
        if len(word) is not 1:
            return False
        if _regex.match(_REPLACE_NO_SPACE, word) is not None:
            return True
        return False

    @staticmethod
    def removePuncInLine(sentence):
        """Remove punctuation in sentence
        @sentence to be edited
        Return the edited string"""
        _FUNC_DEBUG = True
        newSen = ''
        splitted = _regex.compile(_WHITE_SPACE).split(sentence)
        for word in splitted:
            if (_FUNC_DEBUG):
                print('word: ' + word)
            if Process.isPunc(word) is not True:
                newSen += word + ' '
        return newSen.strip()

    def run(self):
        """Function to run and process the movie"""
        print('Running...')
        self.docs = Util.readAllFilesInFolder(
            _POS_FOLDER,
            numFiles=2,
            eachLineCallback=Process.removePuncInLine)

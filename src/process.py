# This file is used to process the movie
# classification, using other classes.

from util.util import Util
from sklearn.feature_extraction.text import CountVectorizer

# Default folder for the positive path
_POS_FOLDER = './assets/review_polarity/txt_sentoken/pos/*.txt'
_REPLACE_NO_SPACE = "[.;:!\'?,\"()\[\]]"


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
        self.docs = Util.readAllFilesInFolder(_POS_FOLDER, numFiles=2)

    # Remove punctuation
    def isPunc(self, word):
        if len(word) == 1:
            return False
        if word.match(_REPLACE_NO_SPACE, case=False):
            return True
        return False

    # Function to run and process the movie
    def run(self):
        print('Running...')

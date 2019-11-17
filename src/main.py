# Author: Vicky Mohammad
# Description: File containing the main to
# to run the program.

# Import my files
import sys
from process import Process
from util.util import Util
from lib.pos import POSTagging
# Import lib
from sklearn import svm
from sklearn import datasets
from matplotlib import pyplot as _matplot
import scipy
import numpy

# For the debug statement
_FILE_EXAMPLE = './assets/review_polarity/txt_sentoken/pos/cv000_29590.txt'
_POS_FOLDER = './assets/review_polarity/txt_sentoken/pos/*.txt'


class Main:
    def __init__(self):
        super().__init__()
        Main.run()

    @staticmethod
    def run():
        Process().run()

    @staticmethod
    def testReadFile():
        print('Running script...')
        Util.readAllFilesInFolder(_POS_FOLDER, numFile=3)
        POSTagging().runExample()


Main()

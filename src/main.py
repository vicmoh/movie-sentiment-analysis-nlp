# Import my files
import sys
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

# Main function
def main():
    print('Running script...')
    Util.readAllFilesInFolder(_POS_FOLDER, numFiles=2)
    POSTagging().runExample()
main()

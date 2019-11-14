# Import my files
import sys
sys.path.append('./src')
sys.path.append('./src/lib')
sys.path.append('./src/util')
from util import Util
from pos import POSTagging, POSData
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
    Util().readAllFilesInFolder(_POS_FOLDER, numFiles=2)
    POSTagging().runExample()
main()

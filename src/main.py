# Import my files
import sys
sys.path.append("./src")
sys.path.append("./src/example")
sys.path.append("./src/util")
from util import Util
from plot import Plot
# Import lib
from sklearn import svm
from sklearn import datasets
from matplotlib import pyplot as _matplot
import scipy
import numpy
# For the debug statement
_FILE_EXAMPLE = "./assets/txt_sentoken/pos/cv000_29590.txt"

# Main function
def main():
    print("Running script...")
    Plot.runExample()
    Util.readFile(_FILE_EXAMPLE)
main()

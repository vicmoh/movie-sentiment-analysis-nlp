# import my files
import sys
sys.path.append("./src")
sys.path.append("./src/example")
from plot import Graph
# import lib
from sklearn import svm
from sklearn import datasets
from matplotlib import pyplot as _matplot
import scipy
import numpy
# for the debug statement
_SHOW_READ_FILE_DEBUG_PRINT = False

# main function
def main():
    print("Running script...")
    Graph.runExample()
main()

# read file from and return the list of string for each lines
def readFile(filePath):
    lines = []
    if filePath == None:
        return lines
    file = open(filePath)
    for line in file:
        lines.append(line)
    file.close()
    if _SHOW_READ_FILE_DEBUG_PRINT:
        print("Reading file: " + filePath)
        for each in lines:
            print(each)
    return lines

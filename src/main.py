import numpy
import scipy
from matplotlib import pyplot as plot
from sklearn import datasets
from sklearn import svm

_SHOW_READ_FILE_DEBUG_PRINT = False


def main():
    print("Running script...")


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

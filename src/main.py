import numpy
import scipy
from matplotlib import pyplot as plot
from sklearn import datasets
from sklearn import svm

# Open file and return list of string
# of each lines


def readFile(filePath):
    lines = []
    if filePath == None:
        return []
    file = open('my_text_file.txt')
    for line in file:
        lines.append(line)
    return lines

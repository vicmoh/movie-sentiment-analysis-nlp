import numpy
import scipy
from matplotlib import pyplot as plot
from sklearn import datasets
from sklearn import svm


def main():
    print("Running script...")


def readFile(filePath):
    lines = []
    if filePath == None:
        return lines
    file = open('my_text_file.txt')
    for line in file:
        lines.append(line)
    file.close()
    return lines

import numpy
import scipy
from matplotlib import pyplot as plot
from sklearn import datasets
from sklearn import svm

plot.xlabel('X Label')
plot.ylabel('Y label')
plot.title('Testing')
plot.plot([1, 2, 3], [6, 7, 8], label = 'First')
plot.plot([1, 2, 3], [5, 6, 7], label = 'Second')
plot.legend() # plot.legend(['First', 'Second])
plot.show()
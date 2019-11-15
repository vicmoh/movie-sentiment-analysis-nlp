# Author: Vicky Mohammad
# Program: Plot file for graphing

# The plots and bars
from matplotlib import pyplot as _matplot


class Plot:
    def __init__(self):
        super().__init__()

    def runExample(self):
        # Init variable
        data1 = [3, 4, 5, 6, 7, 8]
        data2 = [4, 5, 6, 7, 8, 9]
        xData = [1, 2, 3, 4, 5, 6]
        # Graph the model
        _matplot.xlabel('X Label')
        _matplot.ylabel('Y label')
        _matplot.title('Testing')
        _matplot.plot(xData, data1, label='First')
        _matplot.plot(xData, data2, label='Second')
        _matplot.bar(xData, data1, label='Bar', color='grey')
        _matplot.legend()  # _matplot.legend(['First', 'Second])
        _matplot.show()

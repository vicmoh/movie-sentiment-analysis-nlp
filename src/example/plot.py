from matplotlib import pyplot as _matplot

class Plot:
    def __init__(self):
        super().__init__()
    # run matplot example
    def runExample(self):
        # init variable
        data1 = [3, 4, 5, 6, 7, 8]
        data2 = [4, 5, 6, 7, 8, 9]
        xData = [1, 2, 3, 4, 5, 6]
        # graph the model
        _matplot.xlabel('X Label')
        _matplot.ylabel('Y label')
        _matplot.title('Testing')
        _matplot.plot(xData, data1, label='First')
        _matplot.plot(xData, data2, label='Second')
        _matplot.bar(xData, data1, label='Bar', color='grey')
        _matplot.legend()  # _matplot.legend(['First', 'Second])
        _matplot.show()

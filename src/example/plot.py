from matplotlib import pyplot as Graph

data1 = [3, 4, 5, 6, 7, 8]
data2 = [4, 5, 6, 7, 8, 9]
xData = [1, 2, 3, 4, 5, 6]

# Line graph
Graph.xlabel('X Label')
Graph.ylabel('Y label')
Graph.title('Testing')
Graph.plot(xData, data1, label='First')
Graph.plot(xData, data2, label='Second')
Graph.bar(xData, data1, label='Bar', color='grey')
Graph.legend()  # Graph.legend(['First', 'Second])
Graph.show()

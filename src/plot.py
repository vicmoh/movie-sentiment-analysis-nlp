# Author: Vicky Mohammad
# Program: Plot file for graphing.

# The plots and bars
from matplotlib import pyplot as _matplot

_TERM_SCORES = [
    2.1027757996012726,
    1.9672003391113897,
    1.5391921579722103,
    1.4453237610797574,
    1.3575785255597108,
    1.282416532808678,
    1.2682064377459796,
    1.2654283028648234,
    1.2581072461117082,
    1.2467559661676133,
    1.231434212666976,
    1.196765117088466,
    1.1967016137597597,
    1.1798495460912484,
    1.1745258498986815,
    1.1626470874039176,
    1.1576426166065328,
    1.1329569460473297,
    1.1264074553853793,
    1.116517674038553,
]

_TERM_SCORES_NEG = [
    -2.5733553733960823,
    -1.7377615786588994,
    -1.7243384342220458,
    -1.7232987122554924,
    -1.663624292492387,
    -1.6626977547440907,
    -1.659864687097028,
    -1.6029171002927003,
    -1.5687927756471056,
    -1.4973792030405935,
    -1.4707252732755975,
    -1.353577034844943,
    -1.3262626503877692,
    -1.3018523732271154,
    -1.2867487965440023,
    -1.2406475191187047,
    -1.2340224679239145,
    -1.2168412363323506,
    -1.216034334522184,
    -1.1881058509259776
]

_TERM_LABELS = [
    "fun",
    "well",
    "great",
    "hilarious",
    "seen",
    "solid",
    "terrific",
    "different",
    "pulp",
    "good",
    "excellent",
    "memorable",
    "flaw",
    "definitely",
    "many",
    "people",
    "carry",
    "allows",
    "always",
    "life",
]

_TERM_LABELS_NEG = [
    "bad",
    "supposed",
    "awful",
    "boring",
    "nothing",
    "attempt",
    "worst",
    "plot",
    "ridiculous",
    "minute",
    "waste",
    "mess",
    "could",
    "even",
    "material",
    "lame",
    "better",
    "joan",
    "poor",
    "catch"
]


class Plot:
    def __init__(self):
        super().__init__()
        Plot.runTermPosResult()
        Plot.runTermNegResult()

    @staticmethod
    def runTermPosResult():
        # Init variable
        _matplot.title('Top positive terms')
        _matplot.xlabel('Terms')
        _matplot.ylabel('Scores')
        _matplot.xticks(rotation=45)
        _matplot.bar(_TERM_LABELS, _TERM_SCORES, color='green')
        _matplot.show()

    @staticmethod
    def runTermNegResult():
        # Init variable
        _matplot.title('Top negative terms')
        _matplot.xlabel('Terms')
        _matplot.ylabel('Scores')
        _matplot.xticks(rotation=45)
        _matplot.bar(_TERM_LABELS_NEG, _TERM_SCORES_NEG, color='red')
        _matplot.show()

    @staticmethod
    def runExample():
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


Plot()
